param([switch]$Phone, [string]$PythonExe='')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) { $PythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Environment V5 belum tersedia. Gunakan environment Python 3.11 proyek yang sudah disiapkan.' }
& $pythonExe (Join-Path $PSScriptRoot 'prepare_models.py')
if ($LASTEXITCODE -ne 0) { throw 'Bundle model MVP belum siap. Periksa pesan persiapan model.' }
if ($Phone) {
    python (Join-Path $PSScriptRoot 'create_local_cert.py')
    if ($LASTEXITCODE -ne 0) { throw 'Pembuatan sertifikat belum berhasil. Jalankan create_local_cert.py dengan Python lokal yang memiliki cryptography.' }
    & $pythonExe -u (Join-Path $PSScriptRoot 'server.py') --host 0.0.0.0 --port 8443 --http-port 8765 --cert (Join-Path $PSScriptRoot '.local/cert.pem') --key (Join-Path $PSScriptRoot '.local/key.pem')
} else {
    & $pythonExe -u (Join-Path $PSScriptRoot 'server.py')
}
