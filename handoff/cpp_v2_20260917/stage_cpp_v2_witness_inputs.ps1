$ErrorActionPreference = 'Stop'

# Stage only the five immutable inputs for the 2025-08-04 06:45 UTC V2 witness.
# Compute and all outputs remain on Academy HPC; FD-107 is used here only as a read-only source.
$runRoot = '/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/cpp_v2_20260917'
$items = @(
    @('/home/Data_Pool/data/FY/FY4B/AGRI/4KM/2025/20250804/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250804064500_20250804065959_4000M_V0001.HDF', "$runRoot/input/agri/2025/20250804/"),
    @('/home/Data_Pool_3/data/Global_CPP/ERA5/profile/T/2025/era5-temperature-20250804.nc', "$runRoot/input/era5/profile/T/2025/"),
    @('/home/Data_Pool_3/data/Global_CPP/ERA5/profile/R/2025/era5-relative_humidity-20250804.nc', "$runRoot/input/era5/profile/R/2025/"),
    @('/home/Data_Pool_3/data/Global_CPP/ERA5/single/SKT/2025/era5-skin_temperature-20250804.nc', "$runRoot/input/era5/single/SKT/2025/"),
    @('/home/Data_Pool_3/data/Global_CPP/LCCS/MCD12_IGBP_CMG/MCD12_IGBP_2025_Global_CMG.nc', "$runRoot/input/lccs/")
)

foreach ($item in $items) {
    $source, $destination = $item
    & ssh -o BatchMode=yes -o PasswordAuthentication=no -o ConnectTimeout=12 zjnu-hpc "mkdir -p '$destination'"
    if ($LASTEXITCODE -ne 0) { throw "Cannot create Academy destination: $destination" }
    & scp -3 -o BatchMode=yes -o PasswordAuthentication=no -o ConnectTimeout=12 -- "FD-107:$source" "zjnu-hpc:$destination"
    if ($LASTEXITCODE -ne 0) { throw "Transfer failed: $source" }
}
