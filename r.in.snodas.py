#!/usr/bin/env python3
############################################################################
#
# MODULE:       r.in.snodas
#
# AUTHOR(S):    Andrew Wickert
#
# PURPOSE:      Download and import NOAA SNODAS daily snow rasters into GRASS GIS
#
# COPYRIGHT:    (c) 2026 Andrew Wickert
#
#               This program is free software under the GNU General Public
#               License (>=v2). Read the file COPYING that comes with GRASS
#               for details.
#
#############################################################################

#%module
#% description: Download and import NOAA SNODAS daily snow rasters
#% keyword: raster
#% keyword: import
#% keyword: snow
#% keyword: SWE
#% keyword: hydrology
#% keyword: SNODAS
#%end

#%option G_OPT_R_OUTPUT
#%  key: output
#%  label: Output raster basename ({output}_swe_YYYYMMDD, {output}_depth_YYYYMMDD)
#%  required: yes
#%end

#%option
#%  key: variable
#%  type: string
#%  label: Variable(s) to import (comma-separated)
#%  options: swe,depth
#%  answer: swe
#%  required: yes
#%end

#%option
#%  key: start
#%  type: string
#%  label: Start date (YYYY-MM-DD; earliest available: 2003-10-01)
#%  required: yes
#%end

#%option
#%  key: end
#%  type: string
#%  label: End date (YYYY-MM-DD)
#%  required: yes
#%end

#%flag
#%  key: t
#%  description: Register output rasters as space-time raster datasets (strds)
#%end

import os
import io
import gzip
import tarfile
import tempfile
import atexit
import subprocess
import datetime as dt_mod

import numpy as np
import requests

import grass.script as gs

# ---------------------------------------------------------------------------
# SNODAS masked product constants
# ---------------------------------------------------------------------------
_BASE_URL = 'https://noahrsc.noaa.gov/snodas/masked'
_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_FIRST_DATE = dt_mod.date(2003, 10, 1)

# SNODAS masked grid specification (fallback if .Hdr parsing fails)
_NCOLS   = 6935
_NROWS   = 3351
_XMIN    = -124.733333333   # left edge of western pixel
_YMAX    =   52.875000000   # top edge of northern pixel
_CELLSIZE =   0.00833333333 # 30 arcseconds

# Variable codes embedded in SNODAS filenames
_VAR_CODES = {'swe': '1034', 'depth': '1036'}
_VAR_LABEL = {'swe': 'Snow Water Equivalent', 'depth': 'Snow Depth'}

_TMPFILES = []


def cleanup():
    for f in _TMPFILES:
        try:
            os.unlink(f)
        except OSError:
            pass


def _tmpfile(suffix=''):
    p = tempfile.mktemp(suffix=suffix)
    _TMPFILES.append(p)
    return p


def snodas_url(d):
    """Build the SNODAS masked tar URL for a given date."""
    mon = _MONTHS[d.month - 1]
    return '{}/{}/{:02d}_{}/SNODAS_{}.tar'.format(
        _BASE_URL, d.year, d.month, mon, d.strftime('%Y%m%d')
    )


def parse_hdr(hdr_text):
    """Parse SNODAS .Hdr file into a dict of grid parameters."""
    hdr = {}
    for line in hdr_text.splitlines():
        line = line.strip()
        if ':' in line:
            k, _, v = line.partition(':')
            hdr[k.strip().lower()] = v.strip()
    return hdr


def hdr_geotransform(hdr):
    """Extract GDAL GeoTransform from parsed header dict.
    Falls back to hardcoded masked-product values if keys are absent."""
    try:
        xmin = float(hdr.get('minimum x (lower left x)', hdr.get('minimum x', _XMIN)))
        ymax = float(hdr.get('maximum y (upper right y)', hdr.get('maximum y', _YMAX)))
        ncols = int(hdr.get('number of columns', _NCOLS))
        nrows = int(hdr.get('number of rows', _NROWS))
        xmax = float(hdr.get('maximum x (upper right x)', hdr.get('maximum x',
                     xmin + ncols * _CELLSIZE)))
        cs = (xmax - xmin) / ncols
    except (ValueError, KeyError):
        xmin, ymax, ncols, nrows, cs = _XMIN, _YMAX, _NCOLS, _NROWS, _CELLSIZE
    return xmin, ymax, ncols, nrows, cs


def read_snodas_var(tar, var_code):
    """Extract binary data and header for one variable from an open tarfile.

    Returns (float32 numpy array in mm, hdr dict) or (None, None) if not found.
    """
    dat_member = hdr_member = None
    for m in tar.getmembers():
        if var_code in m.name:
            if m.name.endswith('.dat.gz'):
                dat_member = m
            elif m.name.endswith('.Hdr'):
                hdr_member = m

    if dat_member is None:
        return None, None

    # Read binary (big-endian int16)
    with tar.extractfile(dat_member) as f:
        gz_bytes = f.read()
    with gzip.open(io.BytesIO(gz_bytes)) as gz:
        raw_bytes = gz.read()

    # Parse .Hdr for grid geometry
    hdr = {}
    if hdr_member is not None:
        with tar.extractfile(hdr_member) as f:
            hdr = parse_hdr(f.read().decode('latin-1'))

    # scale_factor = 0.001 (meters); raw int16 × 0.001 = meters → ×1000 = mm
    # so: raw int16 value IS already mm
    scale = float(hdr.get('scaling factor', '0.001'))
    nodata_raw = int(float(hdr.get('no data value', '-9999')))

    arr = np.frombuffer(raw_bytes, dtype='>i2').astype(np.float32)
    arr[arr == nodata_raw] = np.nan
    # Convert to mm: raw × scale × 1000
    arr *= scale * 1000.0

    _, ymax, ncols, nrows, _ = hdr_geotransform(hdr)
    arr = arr.reshape(nrows, ncols)
    return arr, hdr


def write_to_grass(arr, hdr, map_name):
    """Write float32 array to a GRASS raster via binary + VRT + gdal_translate + r.import."""
    xmin, ymax, ncols, nrows, cs = hdr_geotransform(hdr)

    # Write float32 binary (native byte order; north-to-south row order)
    tmp_bin = _tmpfile('.bin')
    out = arr.astype(np.float32).copy()
    out[np.isnan(out)] = np.float32(-9999.0)
    out.tofile(tmp_bin)

    # VRT pointing to the binary
    tmp_vrt = _tmpfile('.vrt')
    vrt = (
        '<VRTDataset rasterXSize="{ncols}" rasterYSize="{nrows}">\n'
        '  <SRS>EPSG:4326</SRS>\n'
        '  <GeoTransform>{xmin}, {cs}, 0, {ymax}, 0, -{cs}</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
        '    <RawRasterBand>\n'
        '      <SourceFilename relativeToVRT="0">{binfile}</SourceFilename>\n'
        '      <ImageOffset>0</ImageOffset>\n'
        '      <PixelOffset>4</PixelOffset>\n'
        '      <LineOffset>{line_offset}</LineOffset>\n'
        '    </RawRasterBand>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    ).format(
        ncols=ncols, nrows=nrows,
        xmin=xmin, ymax=ymax, cs=cs,
        binfile=tmp_bin,
        line_offset=ncols * 4
    )
    with open(tmp_vrt, 'w') as f:
        f.write(vrt)

    tmp_tif = _tmpfile('.tif')
    subprocess.run(
        ['gdal_translate', '-of', 'GTiff', '-a_nodata', '-9999', tmp_vrt, tmp_tif],
        check=True, capture_output=True
    )

    gs.run_command('r.import', input=tmp_tif, output=map_name,
                   resample='bilinear', resolution='region', overwrite=True)


def register_strds(base, var, map_date_pairs):
    """Create strds and register daily maps (end = start + 1 day)."""
    gs.run_command('t.create', type='strds', temporaltype='absolute',
                   output=base,
                   title='SNODAS {} {}'.format(_VAR_LABEL[var], base),
                   description='NOAA SNODAS daily {}'.format(_VAR_LABEL[var]),
                   overwrite=True)
    reg_file = _tmpfile('.txt')
    with open(reg_file, 'w') as f:
        for map_name, d in map_date_pairs:
            d_end = d + dt_mod.timedelta(days=1)
            f.write('{}|{}|{}\n'.format(map_name, d.isoformat(), d_end.isoformat()))
    gs.run_command('t.register', input=base, file=reg_file, overwrite=True)
    gs.message("Registered {} maps in strds '{}'.".format(len(map_date_pairs), base))


def main():
    options, flags = gs.parser()
    atexit.register(cleanup)

    output    = options['output']
    variables = [v.strip() for v in options['variable'].split(',') if v.strip()]
    start_str = options['start']
    end_str   = options['end']
    flag_t    = flags['t']

    try:
        start = dt_mod.date.fromisoformat(start_str)
        end   = dt_mod.date.fromisoformat(end_str)
    except ValueError as e:
        gs.fatal("Invalid date: {}".format(e))

    if start < _FIRST_DATE:
        gs.warning("SNODAS begins 2003-10-01; adjusting start date.")
        start = _FIRST_DATE
    if end < start:
        gs.fatal("end date must be >= start date.")

    invalid = [v for v in variables if v not in _VAR_CODES]
    if invalid:
        gs.fatal("Unknown variable(s): {}. Choose from: swe, depth.".format(
            ', '.join(invalid)))

    # Build date list
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += dt_mod.timedelta(days=1)

    gs.message("Downloading SNODAS for {} day(s), variable(s): {}.".format(
        len(dates), ', '.join(variables)))

    # Track (map_name, date) per variable for strds registration
    output_maps = {v: [] for v in variables}
    n_skipped = 0

    for i, d in enumerate(dates):
        gs.percent(i, len(dates), 1)
        url = snodas_url(d)
        date_str = d.strftime('%Y%m%d')

        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            gs.warning("Cannot download {}: {}".format(url, e))
            n_skipped += 1
            continue

        try:
            with tarfile.open(fileobj=io.BytesIO(resp.content)) as tar:
                for var in variables:
                    arr, hdr = read_snodas_var(tar, _VAR_CODES[var])
                    if arr is None:
                        gs.warning("Variable '{}' not found in {} archive.".format(
                            var, date_str))
                        continue
                    map_name = '{}_{}_{}'.format(output, var, date_str)
                    write_to_grass(arr, hdr, map_name)
                    gs.run_command('r.support', map=map_name,
                                   title='SNODAS {} {}'.format(_VAR_LABEL[var], d.isoformat()),
                                   units='mm',
                                   description='NOAA SNODAS {} in mm'.format(_VAR_LABEL[var]))
                    output_maps[var].append((map_name, d))
        except tarfile.TarError as e:
            gs.warning("Cannot read tar for {}: {}".format(date_str, e))
            n_skipped += 1
            continue

    gs.percent(len(dates), len(dates), 1)

    if flag_t:
        for var in variables:
            if output_maps[var]:
                strds_name = '{}_{}'.format(output, var)
                register_strds(strds_name, var, output_maps[var])

    total = sum(len(v) for v in output_maps.values())
    gs.message("Done: {} raster(s) imported, {} date(s) skipped.".format(
        total, n_skipped))
    if total:
        gs.message("Map names: {}_<var>_YYYYMMDD (e.g. {}_swe_{}).".format(
            output, output, dates[0].strftime('%Y%m%d')))


if __name__ == '__main__':
    main()
