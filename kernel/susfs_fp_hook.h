/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * susfs_fp_hook.h - syscall interception by replacing a sys_call_table entry
 */

#ifndef __SUSFS_FP_HOOK_H
#define __SUSFS_FP_HOOK_H

#include <linux/types.h>

struct pt_regs;

/* The kernel's own syscall entry type: on arm64 a table entry is a function that
 * takes the caller's registers.  Kept as our own typedef so this file does not
 * depend on <asm/syscall.h> being reachable. */
typedef long (*susfs_syscall_fn_t)(const struct pt_regs *regs);

struct susfs_fp_hook {
	/* Index into sys_call_table, or -1 when this syscall has no native arm64 entry at
	 * all (see nr_compat).  A hook with nr < 0 sets nr_compat or it has nothing to
	 * patch, which susfs_fp_install() refuses outright. */
	int nr;					/* __NR_* */
	/* Index into compat_sys_call_table; 0 = this hook has no 32-bit entry.
	 * The two tables do NOT share a numbering and the same number is a DIFFERENT
	 * syscall in them: native openat/openat2 are 56/437 (include/uapi/asm-generic/
	 * unistd.h) while the AArch32 side numbers open 5 and openat 322 (generated from
	 * arch/arm/tools/syscall.tbl) - and native 5 is setxattr.  Indexing the compat
	 * table with nr therefore replaces an unrelated 32-bit syscall, which is why every
	 * compat access below goes through this field and never through nr.
	 * There is deliberately no "0 = the same number as nr" shorthand: it would put an
	 * implicit NATIVE number into a compat index, which is the exact mistake this field
	 * exists to prevent, so every compat index is written out instead. */
	int nr_compat;
	const char *name;			/* our wrapper, for logs */
	const char *sym;			/* the kernel's wrapper symbol, for logs */
	susfs_syscall_fn_t wrapper;		/* what we put into the table */
	susfs_syscall_fn_t *orig_slot;		/* where the wrapper reads the original */
	susfs_syscall_fn_t orig;		/* the entry we replaced */
	bool installed;
	/* The 32-bit half.  A 32-bit task never touches sys_call_table, so a hook with no
	 * compat index is simply not there for it - and an entry layer the compat side cannot
	 * reach is exactly why the compat kprobes exist.
	 * nr_compat alone decides whether this side exists: a separate "also patch compat"
	 * bool would be a second way of asking the same question, and the two could disagree
	 * (a hook marked compat whose nr_compat is 0, i.e. with no number to patch with). */
	susfs_syscall_fn_t *orig_slot_compat;	/* where a wrapper reads the compat original */
	susfs_syscall_fn_t orig_compat;		/* the compat entry we replaced */
};

/* Resolves and caches sys_call_table; 0 on the first success. */
int susfs_fp_init(void);

unsigned long susfs_fp_syscall_table(void);

/* Same for the 32-bit table, 0 when this kernel has none: a diagnostic wants to print what
 * an entry holds, and it cannot reach the compat table through the native one. */
unsigned long susfs_fp_compat_syscall_table(void);

int susfs_fp_install(struct susfs_fp_hook *h);
void susfs_fp_remove(struct susfs_fp_hook *h);

/* Blocks until no CPU can still be inside a wrapper; call before the module text
 * may go away (i.e. once, after all entries are restored). */
void susfs_fp_drain(void);

/* Prints table[nr] next to the plain and the .cfi_jt symbol for one entry. */
void susfs_fp_dump_entry(int nr, const char *sym);

/* Prints the first instruction of a wrapper - 'bti c' or nothing. */
void susfs_fp_dump_wrapper(const char *name, const void *fn);


/* Entry/exit bracket for every wrapper body.
 *
 * susfs_fp_remove() restoring a table entry only stops NEW calls.  It says nothing
 * about a task that already picked a wrapper out of the table - and such a task
 * spends nearly all of its time inside orig(regs), i.e. SLEEPING in the filesystem.
 * Sleeping is a quiescent state for synchronize_rcu_tasks(), so that primitive
 * cannot see it: the drain returned with a task parked in wrapper code, rmmod freed
 * the text, and the task came back into it.  Measured on this device as
 *
 *     Unable to handle kernel paging request at virtual address 0000000080080d0a
 *     pc : susfs_opath_openat+0x38/0xb4 [susfs_guard_lkm]
 *
 * - the stadd of n_opath_seen aimed at a user-range address, because the task was
 * running whatever the freed page had turned into.
 *
 * So the count is explicit.  susfs_fp_drain() waits for it to reach zero (phase 1,
 * covers a task blocked in orig), and only then runs the tasks-RCU drain (phase 2,
 * covers a body that decremented the count and is still finishing its own epilogue
 * in module text).
 *
 * susfs_fp_enter() returns false once a drain has started; the wrapper must then
 * call the original directly and must NOT call susfs_fp_leave().  Every path that
 * returns true has to be paired with exactly one susfs_fp_leave().
 */
bool susfs_fp_enter(void);
void susfs_fp_leave(void);
#endif /* __SUSFS_FP_HOOK_H */
