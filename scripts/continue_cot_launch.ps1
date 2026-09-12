param([Parameter(Mandatory=$true)][string]$StateDir)
$ErrorActionPreference = 'Stop'
$remoteRoot = '/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907'
$packName = 'cot_repaired_pack_20260907_v1'
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$taskLock = [System.IO.File]::Open((Join-Path $StateDir 'launch.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
function State([string]$stage, [string]$detail) {
    $record = @{stage=$stage; detail=$detail; time=(Get-Date -Format o); pid=$PID}
    $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $StateDir 'status.json') -Encoding UTF8
    Write-Output ($record | ConvertTo-Json -Compress)
}
function Remote([string]$hostAlias, [string]$command) {
    $result = & ssh -n -T -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 $hostAlias $command
    if ($LASTEXITCODE -ne 0) { throw "SSH $hostAlias failed with code $LASTEXITCODE" }
    return ($result -join "`n")
}
function FileSha256([string]$path) {
    $stream = [System.IO.File]::OpenRead($path)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
try {
    State 'WAITING_PACK' 'Waiting for verified train/validation pack; no GPU job submitted.'
    $deadline = (Get-Date).AddHours(4)
    while ($true) {
        $payload = Remote 'FD-107' 'cat /tmp/cot_repaired_pack_20260907_v1/pack_status.json'
        $pack = $payload | ConvertFrom-Json
        if ($pack.state -eq 'COMPLETE') { break }
        State 'WAITING_PACK' ("Processed {0}/{1}" -f $pack.processed,$pack.total)
        if ((Get-Date) -gt $deadline) { throw 'Pack wait exceeded 4 hours; no training submitted.' }
        $tail = Remote 'FD-107' 'tail -n 15 /tmp/cot_pack_20260907_v1.log'
        if ($tail -match 'Traceback|AssertionError|ValueError') { throw "Pack failed: $tail" }
        Start-Sleep -Seconds 30
    }
    if ($pack.train -ne 11978 -or $pack.validation -ne 6536 -or $pack.test_payloads_read -ne 0) {
        throw 'Unexpected pack cohort or test usage'
    }
    State 'ARCHIVING' 'Packing verified arrays and provenance for transfer.'
    Remote 'FD-107' 'tar -cf /tmp/cot_repaired_pack_20260907_v1.tar -C /tmp cot_repaired_pack_20260907_v1' | Write-Output
    $remoteHash = (Remote 'FD-107' 'sha256sum /tmp/cot_repaired_pack_20260907_v1.tar').Split(' ')[0]
    $localArchive = Join-Path $StateDir ($packName + '.tar')
    State 'DOWNLOADING' 'Copying compact pack from FD-107.'
    & scp -o BatchMode=yes -o ConnectTimeout=15 'FD-107:/tmp/cot_repaired_pack_20260907_v1.tar' $localArchive
    if ($LASTEXITCODE -ne 0) { throw 'FD-107 download failed' }
    if ((FileSha256 $localArchive) -ne $remoteHash) { throw 'Local archive SHA mismatch' }
    State 'UPLOADING' 'Copying compact pack to college server.'
    & scp -o BatchMode=yes -o ConnectTimeout=15 $localArchive "zjnu-hpc:${remoteRoot}/data/${packName}.tar"
    if ($LASTEXITCODE -ne 0) { throw 'College upload failed' }
    $destinationHash = (Remote 'zjnu-hpc' "sha256sum ${remoteRoot}/data/${packName}.tar").Split(' ')[0]
    if ($destinationHash -ne $remoteHash) { throw 'Destination archive SHA mismatch' }
    State 'VERIFYING' 'Extracting verified archive and verifying every array and code hash.'
    Remote 'zjnu-hpc' "test ! -e ${remoteRoot}/data/${packName} && tar -xf ${remoteRoot}/data/${packName}.tar -C ${remoteRoot}/data" | Write-Output
    $submission = Remote 'zjnu-hpc' "/public/home/slfu/miniconda3/envs/swc/bin/python ${remoteRoot}/scripts/submit_cot_once.py"
    $submission | Set-Content -LiteralPath (Join-Path $StateDir 'submitted_job.json') -Encoding UTF8
    $job = ($submission | ConvertFrom-Json).job_id
    State 'SUBMITTED' ("PBS job {0}; GPU gates still required before formal training." -f $job)
} catch {
    State 'FAILED' $_.Exception.Message
    throw
} finally {
    $taskLock.Dispose()
}
