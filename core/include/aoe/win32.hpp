// Author: Paulo R. A. Gilz
//
// The one place <windows.h> is included, and the settings that have to be in
// force when it is.
//
// Why this file exists
// --------------------
// Windows headers are version-gated: every declaration sits behind a check on
// `_WIN32_WINNT`, and anything newer than the value in force is simply not
// declared. The failure is not a clear "your Windows is too old" -- it is a
// pile of "'SRWLOCK' does not name a type" and "'AcquireSRWLockShared' was not
// declared in this scope" from platform_thread.hpp, which reads like the code
// is broken rather than like a missing define.
//
// MinGW-w64 picks a default when nothing sets the value, and *which* default
// depends on the distribution and its version. That is why this built on one
// machine and not on another with no source change between them. Relying on
// the toolchain's default is relying on a number nobody chose.
//
// So the value is set here, in a header every Windows translation unit gets
// before it sees a Windows type, rather than only in the build files. Four
// build routes exist -- Makefile, build.bat, CMake and the by-hand MSVC line in
// the README -- and a fifth (an IDE, a colleague's script) is always possible.
// A flag has to be repeated in all of them and can be forgotten in any; a
// header cannot.
//
// The build files pass -D_WIN32_WINNT=0x0A00 as well. That is deliberate
// belt-and-braces, not a leftover: it documents the requirement where someone
// reading the build reads it, and the guards below make the two agree rather
// than conflict.
//
// What the code actually needs
// ----------------------------
// 0x0600 (Vista) is the true floor, set by the synchronisation primitives in
// platform_thread.hpp: SRWLOCK, AcquireSRWLockShared and
// SleepConditionVariableSRW all appeared there. Nothing in this project needs
// anything newer.
//
// 0x0A00 (Windows 10) is what is set, for two reasons: the GUI requires PySide6
// which has not supported anything older for years, so a lower floor here would
// buy a compatibility the application cannot deliver; and it is the value the
// build was verified against. Anyone who genuinely needs to target older
// Windows can define _WIN32_WINNT before this header or pass -D with a lower
// value -- the guards below defer to it -- and should expect to hear about it
// from platform_thread.hpp if they go below 0x0600.

#pragma once

#if defined(_WIN32)

// Deferred to if already set, by a build file or an earlier include, so a
// deliberate choice elsewhere is never silently overridden.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

// WINVER is the older spelling of the same idea. Some headers still consult it,
// and the two disagreeing produces failures stranger than either alone.
#ifndef WINVER
#define WINVER _WIN32_WINNT
#endif

// Keeps <windows.h> from dragging in the sockets, RPC, OLE and multimedia
// headers, which this project uses none of and which cost real compile time.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Without this, <windows.h> defines min and max as macros, which then break
// every std::min / std::max in every header included afterwards.
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

// A clear message beats a hundred lines of "does not name a type". This fires
// only if someone deliberately lowered the value, since the default above is
// well clear of it.
#if _WIN32_WINNT < 0x0600
#error "aoeViewer needs _WIN32_WINNT >= 0x0600 (Vista): platform_thread.hpp uses SRWLOCK and SleepConditionVariableSRW."
#endif

#endif  // _WIN32
