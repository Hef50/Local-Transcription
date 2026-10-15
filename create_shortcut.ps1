<#
  Creates "Local Transcriber.lnk" in this folder. Windows can pin a shortcut to the
  taskbar or Start (it can't pin run.bat). Re-run this if you move the folder.

  The shortcut carries the app's taskbar ID (System.AppUserModel.ID), the same one the
  app sets on its window, so the running app shows up on the pinned icon instead of as
  a separate Python button. Shortcuts to this app that are already pinned get the ID too.

  Usage:  powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
#>
$AppId = "LocalTranscriber.App"   # must match APP_ID in recpipe/gui.py
$root = $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$appPy = Join-Path $root "app.py"
$icon = Join-Path $root "assets\app.ico"
if (-not (Test-Path $pythonw)) {
    Write-Host "Run setup.ps1 (or run.bat) first so .venv exists." -ForegroundColor Red
    exit 1
}

Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace LocalTranscriber {
    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore {
        [PreserveSig] int GetCount(out uint count);
        [PreserveSig] int GetAt(uint index, out PropertyKey key);
        [PreserveSig] int GetValue(ref PropertyKey key, out PropVariant value);
        [PreserveSig] int SetValue(ref PropertyKey key, ref PropVariant value);
        [PreserveSig] int Commit();
    }

    [ComImport, Guid("0000010b-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPersistFile {
        [PreserveSig] int GetClassID(out Guid classId);
        [PreserveSig] int IsDirty();
        [PreserveSig] int Load([MarshalAs(UnmanagedType.LPWStr)] string fileName, uint mode);
        [PreserveSig] int Save([MarshalAs(UnmanagedType.LPWStr)] string fileName, [MarshalAs(UnmanagedType.Bool)] bool remember);
        [PreserveSig] int SaveCompleted([MarshalAs(UnmanagedType.LPWStr)] string fileName);
        [PreserveSig] int GetCurFile([MarshalAs(UnmanagedType.LPWStr)] out string fileName);
    }

    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    class ShellLink { }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey {
        public Guid FormatId;
        public uint PropertyId;
        public PropertyKey(Guid formatId, uint propertyId) { FormatId = formatId; PropertyId = propertyId; }
    }

    [StructLayout(LayoutKind.Explicit, Size = 24)]
    public struct PropVariant {
        [FieldOffset(0)] public ushort VarType;
        [FieldOffset(8)] public IntPtr Pointer;
    }

    public static class AppUserModel {
        static readonly PropertyKey IdKey = new PropertyKey(new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), 5);
        const ushort VT_LPWSTR = 31;
        const uint STGM_READWRITE = 2;

        public static void SetId(string shortcutPath, string appId) {
            var link = new ShellLink();
            try {
                var file = (IPersistFile)link;
                Marshal.ThrowExceptionForHR(file.Load(shortcutPath, STGM_READWRITE));
                var store = (IPropertyStore)link;
                var key = IdKey;
                var value = new PropVariant { VarType = VT_LPWSTR, Pointer = Marshal.StringToCoTaskMemUni(appId) };
                try {
                    Marshal.ThrowExceptionForHR(store.SetValue(ref key, ref value));
                    Marshal.ThrowExceptionForHR(store.Commit());
                } finally {
                    Marshal.FreeCoTaskMem(value.Pointer);
                }
                Marshal.ThrowExceptionForHR(file.Save(shortcutPath, true));
            } finally {
                Marshal.ReleaseComObject(link);
            }
        }

        public static string GetId(string shortcutPath) {
            var link = new ShellLink();
            try {
                Marshal.ThrowExceptionForHR(((IPersistFile)link).Load(shortcutPath, 0));
                var key = IdKey;
                PropVariant value;
                Marshal.ThrowExceptionForHR(((IPropertyStore)link).GetValue(ref key, out value));
                return value.VarType == VT_LPWSTR ? Marshal.PtrToStringUni(value.Pointer) : null;
            } finally {
                Marshal.ReleaseComObject(link);
            }
        }
    }
}
"@

$shell = New-Object -ComObject WScript.Shell
function Set-AppShortcut([string]$path) {
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = $pythonw
    $shortcut.Arguments = '"' + $appPy + '"'
    $shortcut.WorkingDirectory = $root
    $shortcut.IconLocation = "$icon,0"
    $shortcut.Description = "Local Transcriber: offline transcripts with speaker labels"
    $shortcut.Save()
    [LocalTranscriber.AppUserModel]::SetId($path, $AppId)
}

$path = Join-Path $root "Local Transcriber.lnk"
Set-AppShortcut $path
Write-Host "Created $path  (taskbar ID: $([LocalTranscriber.AppUserModel]::GetId($path)))"

# Shortcuts to this app that are already pinned to the taskbar or Start.
$pinFolders = @(
    (Join-Path $env:APPDATA "Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar"),
    (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs")
)
$updated = 0
foreach ($folder in $pinFolders) {
    if (-not (Test-Path $folder)) { continue }
    Get-ChildItem $folder -Filter *.lnk -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        $existing = $shell.CreateShortcut($_.FullName)
        if ($existing.TargetPath -ieq $pythonw -and $existing.Arguments -like "*app.py*") {
            Set-AppShortcut $_.FullName
            Write-Host "Updated pinned shortcut: $($_.FullName)"
            $updated++
        }
    }
}
if ($updated -eq 0) {
    Write-Host "To pin: right-click 'Local Transcriber' in this folder > Show more options > Pin to taskbar."
} else {
    Write-Host "If the taskbar still shows a separate Python button, unpin the icon and pin the new shortcut once."
}
