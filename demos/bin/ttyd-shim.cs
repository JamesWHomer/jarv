// ttyd shim for VHS recordings (compiled by _record-common.ps1 via the built-in
// .NET Framework csc.exe). VHS always launches ttyd with `--once` ("allow one
// connection and exit"). Its browser opens the ttyd page twice — once on target
// creation, once after the viewport is set — so on Windows the first disconnect
// takes ttyd down before the terminal is ready, and VHS then waits forever for
// an xterm canvas that never renders. This shim sits ahead of the real ttyd on
// PATH and drops `--once`, passing every other argument through verbatim.
//
// Because the shim is the process VHS kills at the end of a take, the real ttyd
// would otherwise outlive it; the child is put in a kill-on-close job object so
// it dies with the shim instead of leaking a PowerShell session per recording.
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

class TtydShim
{
    static int Main()
    {
        string raw = Environment.CommandLine;
        string args;
        if (raw.StartsWith("\""))
        {
            int end = raw.IndexOf('"', 1);
            args = raw.Substring(end + 1);
        }
        else
        {
            int end = raw.IndexOf(' ');
            args = end < 0 ? "" : raw.Substring(end + 1);
        }
        args = args.TrimStart().Replace("--once ", "").Replace(" --once", "");

        string realTtyd = Environment.GetEnvironmentVariable("TTYD_SHIM_REAL");
        if (string.IsNullOrEmpty(realTtyd))
        {
            Console.Error.WriteLine("ttyd-shim: TTYD_SHIM_REAL not set");
            return 1;
        }

        var psi = new ProcessStartInfo(realTtyd, args) { UseShellExecute = false };
        using (var p = Process.Start(psi))
        {
            KillChildWithMe(p);
            p.WaitForExit();
            return p.ExitCode;
        }
    }

    // Job objects are the only cleanup that survives TerminateProcess: no
    // finally block or exit handler runs when VHS kills the shim.
    static void KillChildWithMe(Process child)
    {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) return;

        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        int size = Marshal.SizeOf(info);
        IntPtr ptr = Marshal.AllocHGlobal(size);
        Marshal.StructureToPtr(info, ptr, false);
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, ptr, (uint)size);
        Marshal.FreeHGlobal(ptr);

        AssignProcessToJobObject(job, child.Handle);
    }

    const int JobObjectExtendedLimitInformation = 9;
    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll")]
    static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint infoLength);

    [DllImport("kernel32.dll")]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }
}
