// ffmpeg shim for VHS recordings (compiled by record-all.ps1 via the built-in
// .NET Framework csc.exe). VHS lets ffmpeg pick codec defaults from the output
// extension, so .webp comes out lossy VP8 with 4:2:0 chroma subsampling — bad
// for colored terminal text. This shim sits ahead of the real ffmpeg on PATH,
// upgrades animated-WebP encodes to lossless RGB, and passes everything else
// through untouched. The raw command line is forwarded verbatim (no re-quoting).
using System;
using System.Diagnostics;
using System.IO;

class FfmpegShim
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
        args = args.TrimStart();

        string logPath = Environment.GetEnvironmentVariable("FFMPEG_SHIM_LOG");
        if (!string.IsNullOrEmpty(logPath))
            File.AppendAllText(logPath, args + "\r\n\r\n");

        string realFfmpeg = Environment.GetEnvironmentVariable("FFMPEG_SHIM_REAL");
        if (string.IsNullOrEmpty(realFfmpeg))
        {
            Console.Error.WriteLine("ffmpeg-shim: FFMPEG_SHIM_REAL not set");
            return 1;
        }

        if (args.Contains(".webp"))
            args = RewriteWebp(args);

        var psi = new ProcessStartInfo(realFfmpeg, args) { UseShellExecute = false };
        using (var p = Process.Start(psi))
        {
            p.WaitForExit();
            return p.ExitCode;
        }
    }

    static string RewriteWebp(string args)
    {
        // VHS passes no codec/quality options — the .webp output path is the
        // final token. Insert lossless-RGB encode options just before it.
        // VP8L keeps exact colors (no 4:2:0 chroma subsampling) and compresses
        // flat-color terminal content well.
        int i = args.LastIndexOf(' ');
        if (i < 0) return args;
        return args.Substring(0, i)
            + " -c:v libwebp_anim -lossless 1 -compression_level 4 -loop 0 -pix_fmt bgra"
            + args.Substring(i);
    }
}
