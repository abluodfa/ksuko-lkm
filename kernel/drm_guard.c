/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * drm_guard.c - Widevine "deviceUniqueId" interception, M3 (scoped, opt-in write).
 *
 * Cost control, which is the whole point of M3:
 *
 *   - binder_ioctl(filp, cmd, arg) is called ~1.5M times on this device.  The
 *     command word sits in a register, so everything that is not
 *     BINDER_WRITE_READ leaves after one compare (cmd_skip).
 *   - Of what is left, only write streams of at most DRM_WRITE_MAX bytes can
 *     carry a getPropertyByteArray parcel; bulk streams are dropped without
 *     touching user memory (big_skip).
 *   - The exit hook only scans reads of a tgid that asked for deviceUniqueId
 *     within the last DRM_WATCH_TTL reads (watch_hits / pend_miss), so user
 *     memory is read a few dozen times instead of a million.
 *
 * Recognition rules, carried over from the SKP device identity module (DRMID
 * 1.4.0 / kernel_context.h + binder_hook_builder.cpp):
 *
 *   request : a transaction whose parcel carries "deviceUniqueId"
 *   reply   : BR_REPLY whose parcel starts with status == 0 and a byte-array
 *             length of exactly 32
 *   scope   : tgid whitelist published from user space
 *
 * Where this differs from the original, and why:
 *
 *   - The hook is a kprobe on binder_ioctl with a pre/post pair, not the SKRoot
 *     kernel_module_kit before/after hook, so nothing is written into kernel text.
 *   - The request MUST be parsed in pre_handler: after binder_ioctl returns,
 *     libbinder has already freed or recycled the parcel buffer that
 *     binder_transaction_data.data.ptr.buffer pointed at, so a post-return scan
 *     reads stale memory and never sees the property name.
 *   - The request and its reply are normally in DIFFERENT ioctl calls (the client
 *     receives BR_TRANSACTION_COMPLETE first and blocks for the reply in a later
 *     call), so a per-thread pending table carries the request across calls - the
 *     same reason the original keeps a pending ring.
 *
 * M1.3 is still observation only: counters plus a bounded amount of logging, the
 * 32 bytes are never modified.
 */
#define pr_fmt(fmt) "drm_guard: " fmt

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/sched.h>
#include <linux/uaccess.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/atomic.h>
#include <linux/string.h>
#include <linux/random.h>
#include <linux/types.h>
#include <uapi/linux/android/binder.h>

#include "drm_guard.h"
#include "susfs_kprobe.h"

#define DRM_ID_BYTES 32
#define DRM_SCAN_BYTES 4096
#define DRM_CHUNK_BYTES 256
#define DRM_OVERLAP_BYTES 48
#define DRM_PARCEL_MAX 1024
#define DRM_GET_PROPERTY_BYTE_ARRAY 11
#define DRM_HAL_LIMIT 8
#define DRM_PEND_SLOTS 128
/* Only ioctls whose write stream is small can carry a getPropertyByteArray
 * request; the parcel of one is ~116 bytes in total.  Anything bigger is bulk
 * binder traffic and is dropped without touching user memory. */
#define DRM_WRITE_MAX 640
/* How many binder reads after a request are still worth scanning for its reply. */
#define DRM_WATCH_TTL 256
#ifndef BINDER_WRITE_READ
#define BINDER_WRITE_READ _IOWR('b', 1, struct binder_write_read)
#endif

static const char kDrmDescriptor[] = "android.hardware.drm.IDrmPlugin";
static const char kDeviceUniqueId[] = "deviceUniqueId";

/* One request waiting for its reply, keyed by the binder thread that issued it. */
struct drm_pend {
	u64 task;
	u32 tid;
	u32 code;
	bool armed;
};

struct drm_stats {
	atomic64_t calls;
	atomic64_t pre_calls;
	atomic64_t post_calls;
	atomic64_t write_streams;
	atomic64_t read_streams;
	atomic64_t parse_faults;
	atomic64_t code_hits;
	atomic64_t property_hits;
	atomic64_t descriptor_hits;
	atomic64_t drmish_hits;
	atomic64_t reply_candidates;
	atomic64_t would_replace;
	atomic64_t pend_store;
	atomic64_t pend_match;
	atomic64_t pend_miss;
	atomic64_t wsz_nonzero;
	atomic64_t rcons_nonzero;
	atomic64_t arg_miss;
	atomic64_t replaced;
	atomic64_t first_hit;
	atomic64_t unknown_br;
	atomic64_t reply_seen;
	atomic64_t read_dumps;
	atomic64_t big_skip;
	atomic64_t cmd_skip;
	atomic64_t watch_hits;
	atomic64_t reported;
};

/* Argument registers are scratch by the time the probed function returns, so the
 * entry hook stashes the ioctl's user pointer here for the exit hook to use. */
struct drm_call {
	u64 task;
	u32 tid;
	bool armed;
	unsigned long arg;
};

static struct drm_stats g_stats;
static struct drm_pend g_pend[DRM_PEND_SLOTS];
static struct drm_call g_call[DRM_PEND_SLOTS];
static u32 g_hal_tgids[DRM_HAL_LIMIT];
static u32 g_hal_count;
static u32 g_hal_enforce;
/* Auto-learned watch list.  A process enters it when it asks Widevine for
 * "deviceUniqueId", and stays in it for DRM_WATCH_TTL binder reads - only those
 * reads are scanned for the reply.  Everything else pays one array lookup. */
struct drm_watch {
	u32 tgid;
	u32 ttl;
};
static struct drm_watch g_watch[DRM_HAL_LIMIT];
static DEFINE_SPINLOCK(g_watch_lock);

static void drm_watch_arm(u32 tgid)
{
	unsigned long flags;
	int i, slot = -1;

	spin_lock_irqsave(&g_watch_lock, flags);
	for (i = 0; i < DRM_HAL_LIMIT; i++) {
		if (g_watch[i].tgid == tgid) {
			slot = i;
			break;
		}
		if (!g_watch[i].tgid && slot < 0)
			slot = i;
	}
	if (slot < 0)
		slot = 0;
	g_watch[slot].tgid = tgid;
	g_watch[slot].ttl = DRM_WATCH_TTL;
	spin_unlock_irqrestore(&g_watch_lock, flags);
}

/* Returns true when this tgid is currently watched (and consumes one tick). */
static bool drm_watch_tick(u32 tgid)
{
	unsigned long flags;
	bool watched = false;
	int i;

	spin_lock_irqsave(&g_watch_lock, flags);
	for (i = 0; i < DRM_HAL_LIMIT; i++) {
		if (g_watch[i].tgid != tgid)
			continue;
		watched = true;
		if (g_watch[i].ttl)
			g_watch[i].ttl--;
		else
			g_watch[i].tgid = 0;
		break;
	}
	spin_unlock_irqrestore(&g_watch_lock, flags);
	return watched;
}

/* The value every Widevine client will see once g_replace is turned on. */
static u8 g_virtual_id[DRM_ID_BYTES];
static u32 g_replace;

static int drm_hexval(char c)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;
	if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;
	return -1;
}

/* "64 hex chars" -> 32 bytes. */
static bool drm_parse_vid(const char *hex, u8 *out)
{
	int i;

	for (i = 0; i < DRM_ID_BYTES; i++) {
		int hi = drm_hexval(hex[i * 2]);
		int lo = drm_hexval(hex[i * 2 + 1]);

		if (hi < 0 || lo < 0)
			return false;
		out[i] = (u8)((hi << 4) | lo);
	}
	return true;
}

static bool drm_gate_active(void)
{
	return READ_ONCE(g_hal_enforce) != 0 && READ_ONCE(g_hal_count) != 0;
}

static bool drm_gate_allows(u32 tgid)
{
	u32 count = READ_ONCE(g_hal_count);
	u32 i;

	if (!READ_ONCE(g_hal_enforce))
		return true;
	for (i = 0; i < count && i < DRM_HAL_LIMIT; i++) {
		if (READ_ONCE(g_hal_tgids[i]) == tgid)
			return true;
	}
	return false;
}

static u32 drm_pend_index(u64 task)
{
	u64 h = (task >> 6) ^ (task >> 17) ^ (task >> 29);

	return (u32)h & (DRM_PEND_SLOTS - 1);
}

static void drm_pend_store(u64 task, u32 tid, u32 code)
{
	struct drm_pend *slot = &g_pend[drm_pend_index(task)];

	WRITE_ONCE(slot->armed, false);
	slot->task = task;
	slot->tid = tid;
	slot->code = code;
	WRITE_ONCE(slot->armed, true);
	atomic64_inc(&g_stats.pend_store);
}

/* Consume the pending request of this thread.  A miss is the normal case for the
 * many binder reads that are not DRM property lookups. */
static bool __maybe_unused drm_pend_take(u64 task, u32 tid, u32 *code)
{
	struct drm_pend *slot = &g_pend[drm_pend_index(task)];

	if (!READ_ONCE(slot->armed) || slot->task != task || slot->tid != tid)
		return false;
	*code = slot->code;
	WRITE_ONCE(slot->armed, false);
	atomic64_inc(&g_stats.pend_match);
	return true;
}

static void drm_call_store(u64 task, u32 tid, unsigned long arg)
{
	struct drm_call *slot = &g_call[drm_pend_index(task)];

	WRITE_ONCE(slot->armed, false);
	slot->task = task;
	slot->tid = tid;
	slot->arg = arg;
	WRITE_ONCE(slot->armed, true);
}

static unsigned long drm_call_take(u64 task, u32 tid)
{
	struct drm_call *slot = &g_call[drm_pend_index(task)];
	unsigned long arg;

	if (!READ_ONCE(slot->armed) || slot->task != task || slot->tid != tid)
		return 0;
	arg = slot->arg;
	WRITE_ONCE(slot->armed, false);
	return arg;
}

static bool drm_buf_has_token(const u8 *buf, size_t len, const char *token)
{
	size_t tlen = strlen(token);
	size_t i, j;

	if (len < tlen)
		return false;
	for (i = 0; i + tlen <= len; i++) {
		for (j = 0; j < tlen; j++) {
			if (buf[i + j] != (u8)token[j])
				break;
		}
		if (j == tlen)
			return true;
	}
	if (len < tlen * 2)
		return false;
	for (i = 0; i + tlen * 2 <= len; i++) {
		for (j = 0; j < tlen; j++) {
			if (buf[i + j * 2] != (u8)token[j] ||
			    buf[i + j * 2 + 1] != 0)
				break;
		}
		if (j == tlen)
			return true;
	}
	return false;
}

static bool drm_user_has_token(const void __user *base, size_t size,
			       const char *token)
{
	u8 chunk[DRM_CHUNK_BYTES + DRM_OVERLAP_BYTES];
	size_t cap = min_t(size_t, size, DRM_SCAN_BYTES);
	size_t off = 0;

	if (!base || size < strlen(token))
		return false;
	while (off < cap) {
		size_t n = min_t(size_t, cap - off, sizeof(chunk));

		if (copy_from_user(chunk, (const u8 __user *)base + off, n))
			return false;
		if (drm_buf_has_token(chunk, n, token))
			return true;
		if (n <= DRM_OVERLAP_BYTES)
			break;
		off += n - DRM_OVERLAP_BYTES;
	}
	return false;
}

static u32 drm_bc_record_size(u32 cmd)
{
	switch (cmd) {
	case BC_TRANSACTION:
	case BC_REPLY:
		return sizeof(u32) + sizeof(struct binder_transaction_data);
#ifdef BC_TRANSACTION_SEC_CTX
	case BC_TRANSACTION_SEC_CTX:
		return sizeof(u32) + sizeof(struct binder_transaction_data) +
		       sizeof(binder_uintptr_t);
#endif
	case BC_FREE_BUFFER:
	case BC_DEAD_BINDER_DONE:
		return sizeof(u32) + sizeof(binder_uintptr_t);
	case BC_INCREFS:
	case BC_ACQUIRE:
	case BC_RELEASE:
	case BC_DECREFS:
	case BC_ACQUIRE_RESULT:
		return sizeof(u32) + sizeof(u32);
	case BC_ATTEMPT_ACQUIRE:
		return sizeof(u32) + 2 * sizeof(u32);
	case BC_INCREFS_DONE:
	case BC_ACQUIRE_DONE:
	case BC_REQUEST_DEATH_NOTIFICATION:
	case BC_CLEAR_DEATH_NOTIFICATION:
		return sizeof(u32) + 2 * sizeof(binder_uintptr_t);
	case BC_REGISTER_LOOPER:
	case BC_ENTER_LOOPER:
	case BC_EXIT_LOOPER:
		return sizeof(u32);
	default:
		return 0;
	}
}

struct drm_request_result {
	bool code_match;
	bool property_match;
	bool descriptor_match;
	bool drmish_seen;
	u32 last_code;
	u32 last_data_size;
	u32 drmish_code;
	u32 drmish_size;
};

/* Valid only from pre_handler: the parcel is freed once the ioctl returns. */
static void drm_scan_write_stream(struct binder_write_read *bwr,
				  struct drm_request_result *out)
{
	/* At ioctl entry write_consumed is still zero (the driver fills it in on
	 * the way out), so the pending length of the command stream is write_size. */
	u32 consumed = min_t(u32, READ_ONCE(bwr->write_size), DRM_SCAN_BYTES);
	const u8 __user *stream = (const u8 __user *)bwr->write_buffer;
	u32 off = 0;

	if (!stream)
		return;
	while (off + sizeof(u32) <= consumed) {
		u32 cmd;
		struct binder_transaction_data td;
		u32 step;

		if (copy_from_user(&cmd, stream + off, sizeof(cmd)))
			return;

		if (cmd == BC_TRANSACTION) {
			if (off + sizeof(u32) + sizeof(td) > consumed)
				return;
			if (copy_from_user(&td, stream + off + sizeof(u32),
					   sizeof(td)))
				return;
			if (td.code == DRM_GET_PROPERTY_BYTE_ARRAY)
				out->code_match = true;
			if (td.data_size >= 8 && td.data_size <= DRM_PARCEL_MAX &&
			    td.data.ptr.buffer) {
				const void __user *parcel =
					(const void __user *)td.data.ptr.buffer;
				u32 size = td.data_size;

				if (drm_user_has_token(parcel, size,
						       kDeviceUniqueId)) {
					out->property_match = true;
					out->last_code = td.code;
					out->last_data_size = size;
					if (drm_user_has_token(parcel, size,
							       kDrmDescriptor))
						out->descriptor_match = true;
					return;
				}
				if (!out->drmish_seen &&
				    drm_user_has_token(parcel, size, "drm")) {
					out->drmish_seen = true;
					out->drmish_code = td.code;
					out->drmish_size = size;
				}
			}
		}

		step = drm_bc_record_size(cmd);
		if (!step)
			step = sizeof(u32);	/* resynchronise, never give up */
		if (off + step <= off)
			return;
		off += step;
	}
}

/* Walk the read stream (BR_* commands).  A Widevine reply for
 * getPropertyByteArray("deviceUniqueId") is laid out as
 *     u32 status(0) | u32 length(32) | u8 id[32]
 * so the payload is exactly 40 bytes and the id starts at +8. */
static bool drm_scan_read_stream(struct binder_write_read *bwr)
{
	/* Valid at exit: the driver has already written the reply stream. */
	u32 consumed = min_t(u32, READ_ONCE(bwr->read_consumed), DRM_SCAN_BYTES);
	const u8 __user *stream = (const u8 __user *)bwr->read_buffer;
	u32 off = 0;

	if (!stream)
		return false;

	/* Diagnostics: what does a populated read stream actually look like? */
	if (consumed && atomic64_inc_return(&g_stats.read_dumps) <= 3) {
		u8 head[48];
		u32 n = min_t(u32, consumed, sizeof(head));

		if (!copy_from_user(head, stream, n))
			pr_info("read stream %u bytes: %*phN\n", consumed, n,
				head);
	}
	while (off + sizeof(u32) <= consumed) {
		u32 cmd;
		u32 step;
		bool candidate = false;
		struct binder_transaction_data td;

		if (copy_from_user(&cmd, stream + off, sizeof(cmd)))
			return false;

		switch (cmd) {
		case BR_REPLY:
		case BR_TRANSACTION:
			if (off + sizeof(u32) + sizeof(td) > consumed)
				return false;
			if (copy_from_user(&td, stream + off + sizeof(u32),
					   sizeof(td)))
				return false;
			step = sizeof(u32) + sizeof(td);
			if (cmd == BR_REPLY &&
			    atomic64_inc_return(&g_stats.reply_seen) <= 6)
				pr_info("BR_REPLY tgid=%d data_size=%u\n",
					current->tgid, td.data_size);
			if (cmd == BR_REPLY &&
			    td.data_size >= DRM_ID_BYTES + 8 &&
			    td.data_size <= DRM_ID_BYTES + 8 + 32 &&
			    td.data.ptr.buffer) {
				void __user *payload =
					(void __user *)td.data.ptr.buffer;
				u32 hdr[2];

				if (copy_from_user(hdr, payload, sizeof(hdr)))
					return false;
				if (hdr[0] == 0 && hdr[1] == DRM_ID_BYTES) {
					candidate = true;
					/* Another request may follow the same way. */
					drm_watch_arm(current->tgid);
					if (READ_ONCE(g_replace)) {
						u8 __user *dst =
							(u8 __user *)payload + 8;

						if (!copy_to_user(
							    dst,
							    g_virtual_id,
							    DRM_ID_BYTES)) {
							atomic64_inc(
							  &g_stats.replaced);
							if (!atomic64_read(
							      &g_stats.first_hit))
								atomic64_set(
								  &g_stats.first_hit,
								  1);
						}
					}
				}
			}
			break;
		case BR_INCREFS:
		case BR_ACQUIRE:
		case BR_RELEASE:
		case BR_DECREFS:
			step = sizeof(u32) + 2 * sizeof(binder_uintptr_t);
			break;
		case BR_DEAD_BINDER:
		case BR_CLEAR_DEATH_NOTIFICATION_DONE:
			step = sizeof(u32) + sizeof(binder_uintptr_t);
			break;
		case BR_NOOP:
		case BR_TRANSACTION_COMPLETE:
		case BR_DEAD_REPLY:
		case BR_FAILED_REPLY:
		case BR_SPAWN_LOOPER:
			step = sizeof(u32);
			break;
		default:
			/* Newer BR codes we do not model: skip the word and keep
			 * walking instead of abandoning the whole stream. */
			if (atomic64_inc_return(&g_stats.unknown_br) <= 8)
				pr_info("unknown BR 0x%x at off=%u\n", cmd, off);
			step = sizeof(u32);
			break;
		}
		if (candidate)
			return true;
		if (step == 0 || off + step <= off)
			return false;
		off += step;
	}
	return false;
}

/* Entry: the parcel is still alive here, so this is the only place a request can
 * be recognised. */
static int drm_pre_handler(struct kprobe *kp, struct pt_regs *regs)
{
	struct binder_write_read bwr;
	struct drm_request_result req = { };
	unsigned long cmd = regs->regs[1];
	unsigned long arg = regs->regs[2];
	(void)kp;
	/* binder_ioctl(filp, cmd, arg).  The command word is a register, so the
	 * whole system pays one compare here and nothing else. */
	if ((u32)cmd != (u32)BINDER_WRITE_READ) {
		atomic64_inc(&g_stats.cmd_skip);
		return 0;
	}
	atomic64_inc(&g_stats.calls);
	if (!arg)
		return 0;
	if (drm_gate_active() && !drm_gate_allows(current->tgid))
		return 0;
	/* Remember the user pointer: x2 is clobbered long before the function
	 * returns, so the exit hook cannot read it from regs. */
	drm_call_store((u64)(uintptr_t)current, current->pid, arg);
	if (copy_from_user(&bwr, (void __user *)arg, sizeof(bwr))) {
		atomic64_inc(&g_stats.parse_faults);
		return 0;
	}
	atomic64_inc(&g_stats.pre_calls);
	/* write_size / write_buffer are inputs; write_consumed is an output the
	 * driver fills in, so it is still zero here and must not be tested. */
	if (!bwr.write_size || !bwr.write_buffer)
		return 0;
	/* A getPropertyByteArray("deviceUniqueId") request is ~116 bytes in total.
	 * Anything larger is bulk traffic: drop it without reading user memory. */
	if (bwr.write_size > DRM_WRITE_MAX) {
		atomic64_inc(&g_stats.big_skip);
		return 0;
	}
	atomic64_inc(&g_stats.write_streams);
	drm_scan_write_stream(&bwr, &req);

	if (req.code_match)
		atomic64_inc(&g_stats.code_hits);
	if (req.descriptor_match)
		atomic64_inc(&g_stats.descriptor_hits);
	if (req.drmish_seen &&
	    atomic64_inc_return(&g_stats.drmish_hits) <= 6)
		pr_info("drm-ish tx tgid=%d code=%u size=%u\n",
			current->tgid, req.drmish_code, req.drmish_size);
	if (req.property_match) {
		if (atomic64_inc_return(&g_stats.property_hits) <= 6)
			pr_info("PROPERTY HIT tgid=%d code=%u size=%u desc=%d\n",
				current->tgid, req.last_code,
				req.last_data_size,
				req.descriptor_match ? 1 : 0);
		drm_pend_store((u64)(uintptr_t)current, current->pid,
			       req.last_code);
		/* From now on this process's binder reads are worth scanning: the
		 * reply to this very request is one of them. */
		drm_watch_arm(current->tgid);
	}
	return 0;
}

/* Return: the reply the kernel just wrote is in the caller's buffer. */
static void drm_post_handler(struct kprobe *kp, struct pt_regs *regs,
			     unsigned long flags)
{
	struct binder_write_read bwr;
	unsigned long arg;

	(void)kp;
	(void)regs;
	if (drm_gate_active() && !drm_gate_allows(current->tgid))
		return;
	/* Only a process that just asked for deviceUniqueId is scanned; every
	 * other reader leaves through here without touching user memory. */
	if (!drm_watch_tick(current->tgid)) {
		atomic64_inc(&g_stats.pend_miss);
		return;
	}
	atomic64_inc(&g_stats.watch_hits);

	arg = drm_call_take((u64)(uintptr_t)current, current->pid);
	if (!arg) {
		atomic64_inc(&g_stats.arg_miss);
		return;
	}

	if (copy_from_user(&bwr, (void __user *)arg, sizeof(bwr))) {
		atomic64_inc(&g_stats.parse_faults);
		return;
	}

	atomic64_inc(&g_stats.post_calls);
	if (bwr.read_consumed)
		atomic64_inc(&g_stats.rcons_nonzero);
	if (!bwr.read_size || !bwr.read_consumed)
		return;

	atomic64_inc(&g_stats.read_streams);
	if (!drm_scan_read_stream(&bwr))
		return;

	atomic64_inc(&g_stats.reply_candidates);
	if (!READ_ONCE(g_replace)) {
		atomic64_inc(&g_stats.would_replace);
		if (atomic64_inc_return(&g_stats.reported) <= 6)
			pr_info("DRM ID reply seen tgid=%d (dry-run, nothing written)\n",
				current->tgid);
		return;
	}
	if (atomic64_inc_return(&g_stats.reported) <= 6)
		pr_info("DRM ID replaced tgid=%d\n", current->tgid);
}

static struct kprobe g_drm_kp = {
	.pre_handler = drm_pre_handler,
	.post_handler = drm_post_handler,
};

static const char *const g_drm_probe_symbols[] = {
	"binder_ioctl",
	"binder_ioctl_write_read",
};

static int drm_probe_arm(void)
{
	size_t i;
	int ret = -ENOENT;

	for (i = 0; i < ARRAY_SIZE(g_drm_probe_symbols); i++) {
		g_drm_kp.symbol_name = g_drm_probe_symbols[i];
		g_drm_kp.addr = NULL;
		ret = register_kprobe(&g_drm_kp);
		if (!ret) {
			pr_info("kprobe armed on %s\n", g_drm_kp.symbol_name);
			return 0;
		}
		g_drm_kp.addr = NULL;
	}
	pr_warn("no probeable binder entry (%d)\n", ret);
	return ret;
}

static void drm_probe_disarm(void)
{
	if (g_drm_kp.symbol_name)
		unregister_kprobe(&g_drm_kp);
	/* Same statement as susfs_kp_forget_addr(), the helper the rest of the
	 * module uses for a probe that was armed once (see susfs_kprobe.h). */
	susfs_kp_forget_addr(&g_drm_kp);
}

static int drm_proc_show(struct seq_file *m, void *v)
{
	seq_printf(m, "mode=%s\n", READ_ONCE(g_replace) ? "write" : "dry-run");
	{
		int i;

		seq_puts(m, "virtual_id=");
		for (i = 0; i < DRM_ID_BYTES; i++)
			seq_printf(m, "%02x", g_virtual_id[i]);
		seq_puts(m, "\n");
	}
	seq_printf(m, "probe=%s\n",
		   g_drm_kp.symbol_name ? g_drm_kp.symbol_name : "off");
	seq_printf(m, "hal_enforce=%u hal_count=%u\n",
		   READ_ONCE(g_hal_enforce), READ_ONCE(g_hal_count));
	{
		unsigned long flags;
		int i;

		seq_puts(m, "watch=");
		spin_lock_irqsave(&g_watch_lock, flags);
		for (i = 0; i < DRM_HAL_LIMIT; i++) {
			if (!g_watch[i].tgid)
				continue;
			seq_printf(m, "%u/%u ", g_watch[i].tgid,
				   g_watch[i].ttl);
		}
		spin_unlock_irqrestore(&g_watch_lock, flags);
		seq_puts(m, "\n");
	}
	seq_printf(m, "cmd_skip=%lld\n", atomic64_read(&g_stats.cmd_skip));
	seq_printf(m, "big_skip=%lld\n", atomic64_read(&g_stats.big_skip));
	seq_printf(m, "watch_hits=%lld\n",
		   atomic64_read(&g_stats.watch_hits));
	seq_printf(m, "calls=%lld\n", atomic64_read(&g_stats.calls));
	seq_printf(m, "pre_calls=%lld\n", atomic64_read(&g_stats.pre_calls));
	seq_printf(m, "post_calls=%lld\n", atomic64_read(&g_stats.post_calls));
	seq_printf(m, "write_streams=%lld\n",
		   atomic64_read(&g_stats.write_streams));
	seq_printf(m, "read_streams=%lld\n",
		   atomic64_read(&g_stats.read_streams));
	seq_printf(m, "parse_faults=%lld\n",
		   atomic64_read(&g_stats.parse_faults));
	seq_printf(m, "code_hits=%lld\n", atomic64_read(&g_stats.code_hits));
	seq_printf(m, "property_hits=%lld\n",
		   atomic64_read(&g_stats.property_hits));
	seq_printf(m, "descriptor_hits=%lld\n",
		   atomic64_read(&g_stats.descriptor_hits));
	seq_printf(m, "drmish_hits=%lld\n",
		   atomic64_read(&g_stats.drmish_hits));
	seq_printf(m, "reply_candidates=%lld\n",
		   atomic64_read(&g_stats.reply_candidates));
	seq_printf(m, "pend_store=%lld\n", atomic64_read(&g_stats.pend_store));
	seq_printf(m, "pend_match=%lld\n", atomic64_read(&g_stats.pend_match));
	seq_printf(m, "pend_miss=%lld\n", atomic64_read(&g_stats.pend_miss));
	seq_printf(m, "rcons_nonzero=%lld\n",
		   atomic64_read(&g_stats.rcons_nonzero));
	seq_printf(m, "arg_miss=%lld\n", atomic64_read(&g_stats.arg_miss));
	seq_printf(m, "replaced=%lld\n", atomic64_read(&g_stats.replaced));
	seq_printf(m, "reply_seen=%lld\n", atomic64_read(&g_stats.reply_seen));
	seq_printf(m, "unknown_br=%lld\n", atomic64_read(&g_stats.unknown_br));
	seq_printf(m, "would_replace=%lld\n",
		   atomic64_read(&g_stats.would_replace));
	return 0;
}

static int drm_proc_open(struct inode *inode, struct file *file)
{
	return single_open(file, drm_proc_show, NULL);
}

static ssize_t drm_proc_write(struct file *file, const char __user *ubuf,
			      size_t count, loff_t *ppos)
{
	char buf[80];
	size_t len = min_t(size_t, count, sizeof(buf) - 1);
	u32 tgid = 0;
	u32 i;

	if (copy_from_user(buf, ubuf, len))
		return -EFAULT;
	buf[len] = '\0';
	strim(buf);

	if (sysfs_streq(buf, "write") || sysfs_streq(buf, "on")) {
		WRITE_ONCE(g_replace, 1);
		pr_info("replacement ON\n");
		return count;
	}
	if (sysfs_streq(buf, "dry") || sysfs_streq(buf, "dry-run")) {
		WRITE_ONCE(g_replace, 0);
		pr_info("replacement OFF (dry-run)\n");
		return count;
	}
	if (!strncmp(buf, "vid ", 4)) {
		u8 tmp[DRM_ID_BYTES];
		const char *hex = buf + 4;

		if (strlen(hex) < DRM_ID_BYTES * 2 ||
		    !drm_parse_vid(hex, tmp))
			return -EINVAL;
		memcpy(g_virtual_id, tmp, DRM_ID_BYTES);
		pr_info("virtual id updated\n");
		return count;
	}
	if (sysfs_streq(buf, "off") || sysfs_streq(buf, "reset")) {
		WRITE_ONCE(g_hal_enforce, 0);
		WRITE_ONCE(g_hal_count, 0);
		memset(g_hal_tgids, 0, sizeof(g_hal_tgids));
		return count;
	}
	if (sscanf(buf, "hal %u", &tgid) != 1)
		return -EINVAL;
	if (tgid == 0) {
		WRITE_ONCE(g_hal_enforce, 0);
		return count;
	}
	for (i = 0; i < DRM_HAL_LIMIT; i++) {
		if (READ_ONCE(g_hal_tgids[i]) == tgid)
			return count;
	}
	for (i = 0; i < DRM_HAL_LIMIT; i++) {
		if (!READ_ONCE(g_hal_tgids[i])) {
			WRITE_ONCE(g_hal_tgids[i], tgid);
			WRITE_ONCE(g_hal_count, READ_ONCE(g_hal_count) + 1);
			WRITE_ONCE(g_hal_enforce, 1);
			pr_info("tgid %u published (%u total)\n", tgid, i + 1);
			return count;
		}
	}
	return -ENOSPC;
}

static const struct proc_ops drm_proc_ops = {
	.proc_open = drm_proc_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
	.proc_write = drm_proc_write,
};

static struct proc_dir_entry *g_drm_proc;

int drm_guard_init(void)
{
	int ret;

	memset(&g_stats, 0, sizeof(g_stats));
	memset(&g_pend, 0, sizeof(g_pend));
	memset(g_call, 0, sizeof(g_call));
	memset(g_hal_tgids, 0, sizeof(g_hal_tgids));
	memset(g_watch, 0, sizeof(g_watch));

	/* A fresh random identity unless user space installs a fixed one. */
	get_random_bytes(g_virtual_id, DRM_ID_BYTES);
	pr_info("virtual id %*phN (dry-run until 'echo write')\n",
		DRM_ID_BYTES, g_virtual_id);

	ret = drm_probe_arm();
	if (ret)
		return ret;

	g_drm_proc = proc_create("susfs_drm_guard", 0644, NULL, &drm_proc_ops);
	if (!g_drm_proc)
		pr_warn("proc entry creation failed; counters unavailable\n");
	return 0;
}

void drm_guard_exit(void)
{
	if (g_drm_proc) {
		proc_remove(g_drm_proc);
		g_drm_proc = NULL;
	}
	drm_probe_disarm();
}