param(
    [switch]$IncludeFileNames
)

$ErrorActionPreference = 'Stop'

# Read-only public metadata query. This script never authenticates, orders,
# downloads, or opens satellite payloads.
$portalBase = 'https://data.nsmc.org.cn/DataPortal/v1/data/selection/subfile'
$productId = 'FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_YYYYMMDDhhmmss_YYYYMMDDhhmmss_4000M_Vnnnn.HDF'
$listingScriptUrl = 'https://data.nsmc.org.cn/DataPortal/js/page.data.file.list.js?v=1.0.250314'

$gapStarts = @(
    '202606110815', '202606110830', '202606192045', '202606192100',
    '202606202130', '202606212200', '202606212215', '202606222245',
    '202606222300', '202606230000', '202607022015', '202607040815',
    '202607050815', '202607060900', '202607070945', '202607131200'
)
$utcDates = @($gapStarts | ForEach-Object { $_.Substring(0, 8) } | Sort-Object -Unique) + @('20260613', '20260614', '20260615')
$utcDates = @($utcDates | Sort-Object -Unique)

$allRows = [System.Collections.Generic.List[object]]::new()
$daily = [System.Collections.Generic.List[object]]::new()

foreach ($dateKey in $utcDates) {
    $date = [datetime]::ParseExact($dateKey, 'yyyyMMdd', [Globalization.CultureInfo]::InvariantCulture).ToString('yyyy-MM-dd')
    $query = [ordered]@{
        productID = $productId
        txtBeginDate = $date
        txtBeginTime = '00:00:00'
        txtEndDate = $date
        txtEndTime = '23:59:59'
        east_CoordValue = 180
        west_CoordValue = -180
        north_CoordValue = 90
        south_CoordValue = -90
        cbAllArea = 'on'
        cbGHIArea = 'on'
        cbSliceArea = ''
        timeSelection = 'all'
        periodTime = ''
        daynight = ''
        where = ''
        sliceAreas = ''
        converStatus = ''
        rdbIsEvery = ''
        beginIndex = 1
        endIndex = 100
    }
    $queryString = ($query.GetEnumerator() | ForEach-Object {
        [uri]::EscapeDataString([string]$_.Key) + '=' + [uri]::EscapeDataString([string]$_.Value)
    }) -join '&'
    $response = Invoke-RestMethod -Method Get -Uri ($portalBase + '?' + $queryString)
    if ($response.status -ne 1) { throw "NSMC metadata query failed for $date (status=$($response.status))" }

    $names = @($response.resource | ForEach-Object { [string]$_.ARCHIVENAME } | Sort-Object)
    $canonical = $names -join "`n"
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = [Convert]::ToHexString($sha.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($canonical))).ToLowerInvariant()
    } finally { $sha.Dispose() }

    $allRows.AddRange(@($response.resource))
    $daily.Add([pscustomobject]@{
        utc_date = $date
        subfile_records = $names.Count
        sorted_archive_name_list_sha256 = $digest
        target_gap_keys = @($gapStarts | Where-Object { $_.StartsWith($dateKey) })
        exact_gap_key_matches = @($gapStarts | Where-Object {
            $key = $_
            $names | Where-Object { $_ -match ("_" + [regex]::Escape($key) + '00_') }
        })
        archive_names = $(if ($IncludeFileNames) { $names } else { @() })
    })
}

$foundStarts = @($allRows | ForEach-Object {
    $m = [regex]::Match([string]$_.ARCHIVENAME, '_(\d{14})_\d{14}_4000M_')
    if ($m.Success) { $m.Groups[1].Value.Substring(0, 12) }
} | Sort-Object -Unique)
$gapMatches = @($gapStarts | Where-Object { $_ -in $foundStarts })

$bjtWindows = foreach ($bjtDate in @('20260614', '20260615')) {
    $date = [datetime]::ParseExact($bjtDate, 'yyyyMMdd', [Globalization.CultureInfo]::InvariantCulture)
    # BJT is UTC+8; use arithmetic on the naive date label and do not invoke
    # workstation-local ToUniversalTime conversions on UTC filename strings.
    $utcStart = $date.AddHours(-8)
    $expected = @(0..95 | ForEach-Object { $utcStart.AddMinutes(15 * $_).ToString('yyyyMMddHHmmss') })
    $listed = @($expected | Where-Object { $_.Substring(0, 12) -in $foundStarts })
    $missing = @($expected | Where-Object { $_.Substring(0, 12) -notin $foundStarts })
    [pscustomobject]@{
        frozen_test_date_bjt = $date.ToString('yyyy-MM-dd')
        utc_window_start = $utcStart.ToString('yyyy-MM-dd HH:mm:ss')
        utc_window_last_start = $utcStart.AddMinutes(15 * 95).ToString('yyyy-MM-dd HH:mm:ss')
        listed_slots = $listed.Count
        expected_slots = 96
        missing_utc_starts = $missing
        missing_bjt_starts = @($missing | ForEach-Object { ([datetime]::ParseExact($_, 'yyyyMMddHHmmss', [Globalization.CultureInfo]::InvariantCulture)).AddHours(8).ToString('yyyy-MM-dd HH:mm:ss') })
    }
}

[pscustomobject]@{
    audit_time_utc = [datetime]::UtcNow.ToString('o')
    scope = 'Public FY-4B AGRI L1 Full Disk 4KM filename metadata only'
    portal_endpoint = $portalBase
    product_id = $productId
    query_contract = $query
    listing_script_url = $listingScriptUrl
    daily = @($daily)
    trainval_gap_keys = $gapStarts
    trainval_gap_exact_matches = $gapMatches
    bjt_test_windows = @($bjtWindows)
    evidence_boundary = 'No authentication, cart/order, download, or satellite-payload access; metadata does not prove downloadable bytes or valid Hubei pixels.'
} | ConvertTo-Json -Depth 8
