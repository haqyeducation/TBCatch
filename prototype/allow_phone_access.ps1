# Run manually as Administrator only if Windows Firewall blocks the phone.
$ErrorActionPreference = 'Stop'
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw 'Buka PowerShell sebagai Administrator, lalu jalankan skrip ini kembali. Skrip hanya mengizinkan TCP 8443 dari jaringan lokal.' }
$ruleName = 'NapasMvpHttpsLocal'
if (-not (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name $ruleName -DisplayName 'TBCatch MVP HTTPS - jaringan lokal' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443 -RemoteAddress LocalSubnet -Profile Private,Public | Out-Null
}
Write-Host 'TCP 8443 diizinkan dari LocalSubnet. Firewall lain tetap aktif.'
Write-Host "Untuk mencabut akses: Remove-NetFirewallRule -Name '$ruleName'"
