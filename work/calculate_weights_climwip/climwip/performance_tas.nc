ЙHDF

                    €€€€€€€€ту      €€€€€€€€        `                               OHDRа"            €€€€€€€€€€€€€€€€€€€€€€€€
     !            model_ensembleK           €€€€€€€€€€€€€€€€€€€€€€€€J        _NCProperties      "          version=2,netcdf=4.7.4,hdf5=1.10.5    ≤      %        йю;OHDRO                                                                                   £      ё            €€€€€€€€€€€€€€€€€€€€€€€€0        CLASS                DIMENSION_SCALE /       NAME                 model_ensemble 4       _Netcdf4Dimid                             ["ЕOHDR                            ?      @ 4 4€      „      ћ                   ш                       €€€€€€€€€€€€€€€€€€€€€€€€P        _FillValue       ?      @ 4 4€                               ш 
              Є™nОOCHK    0      ¬џ            №зzAOCHK            ш+       short_name                   tas!       units                KP       DIMENSION_LIST                                              o™†еOCHK                   Є     t  REFERENCE_LIST       dataset                                       dimension                                                                      ¶              •±r•                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               GCOL                                !       ACCESS1-0_r1i1p1_historical-rcp85              !       ACCESS1-0_r1i1p1_historical-rcp45                     K              X                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              !             !             Z?Taё2р?ј:vБн+у?OCHK           dtas¶      цЏ      provenance         ќЏ         <?xml version='1.0' encoding='ASCII'?>
<prov:document xmlns:file="https://www.esmvaltool.org/file" xmlns:attribute="https://www.esmvaltool.org/attribute" xmlns:preprocessor="https://www.esmvaltool.org/preprocessor" xmlns:task="https://www.esmvaltool.org/task" xmlns:software="https://www.esmvaltool.org/software" xmlns:recipe="https://www.esmvaltool.org/recipe" xmlns:author="https://www.esmvaltool.org/author" xmlns:project="https://www.esmvaltool.org/project" xmlns:prov="http://www.w3.org/ns/prov#" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <prov:activity prov:id="software:esmvaltool==2.1.0"/>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="recipe:recipe_climwip.yml"/>
    <prov:agent prov:ref="author:Brunner, Lukas"/>
  </prov:wasAttributedTo>
  <prov:agent prov:id="author:Kalverla, Peter">
    <attribute:institute>NLeSC, Netherlands</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0002-5025-7862</attribute:orcid>
  </prov:agent>
  <prov:activity prov:id="task:calculate_weights_climwip/climwip"/>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="recipe:recipe_climwip.yml"/>
    <prov:agent prov:ref="author:Kalverla, Peter"/>
  </prov:wasAttributedTo>
  <prov:agent prov:id="author:Brunner, Lukas">
    <attribute:institute>ETH Zurich, Switzerland</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0001-5760-4524</attribute:orcid>
  </prov:agent>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="recipe:recipe_climwip.yml"/>
    <prov:agent prov:ref="author:Smeets, Stef"/>
  </prov:wasAttributedTo>
  <prov:wasStartedBy>
    <prov:activity prov:ref="task:calculate_weights_climwip/climwip"/>
    <prov:trigger prov:ref="recipe:recipe_climwip.yml"/>
    <prov:starter prov:ref="software:esmvaltool==2.1.0"/>
  </prov:wasStartedBy>
  <prov:entity prov:id="recipe:recipe_climwip.yml">
    <attribute:description>EUCP ClimWIP</attribute:description>
    <attribute:references>['brunner2019', 'lorenz2018', 'knutti2017']</attribute:references>
  </prov:entity>
  <prov:agent prov:id="author:Camphuijsen, Jaro">
    <attribute:institute>NLeSC, Netherlands</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0002-8928-7831</attribute:orcid>
  </prov:agent>
  <prov:agent prov:id="author:Smeets, Stef">
    <attribute:institute>NLeSC, Netherlands</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0002-5413-9038</attribute:orcid>
  </prov:agent>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="recipe:recipe_climwip.yml"/>
    <prov:agent prov:ref="author:Camphuijsen, Jaro"/>
  </prov:wasAttributedTo>
  <prov:entity prov:id="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc">
    <attribute:caption>Performance metric (RMS error) for variable tas</attribute:caption>
    <attribute:domains>('regional',)</attribute:domains>
    <attribute:obs_data>native6</attribute:obs_data>
    <attribute:references>['brunner2019', 'lorenz2018', 'knutti2017']</attribute:references>
    <attribute:script>climwip</attribute:script>
    <attribute:script_file>weighting/climwip.py</attribute:script_file>
    <attribute:shape_params>{'tas': {'sigma_d': 0.588, 'sigma_s': 0.704}, 'pr': {'sigma_d': 0.658, 'sigma_s': 0.704}}</attribute:shape_params>
  </prov:entity>
  <prov:agent prov:id="author:Kalverla, Peter">
    <attribute:institute>NLeSC, Netherlands</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0002-5025-7862</attribute:orcid>
  </prov:agent>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc"/>
    <prov:agent prov:ref="author:Kalverla, Peter"/>
  </prov:wasAttributedTo>
  <prov:agent prov:id="author:Smeets, Stef">
    <attribute:institute>NLeSC, Netherlands</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0002-5413-9038</attribute:orcid>
  </prov:agent>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc"/>
    <prov:agent prov:ref="author:Smeets, Stef"/>
  </prov:wasAttributedTo>
  <prov:agent prov:id="author:Brunner, Lukas">
    <attribute:institute>ETH Zurich, Switzerland</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0001-5760-4524</attribute:orcid>
  </prov:agent>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc"/>
    <prov:agent prov:ref="author:Brunner, Lukas"/>
  </prov:wasAttributedTo>
  <prov:agent prov:id="author:Camphuijsen, Jaro">
    <attribute:institute>NLeSC, Netherlands</attribute:institute>
    <attribute:orcid>https://orcid.org/0000-0002-8928-7831</attribute:orcid>
  </prov:agent>
  <prov:wasAttributedTo>
    <prov:entity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc"/>
    <prov:agent prov:ref="author:Camphuijsen, Jaro"/>
  </prov:wasAttributedTo>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2012_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2009_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2001_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:43:38 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2001, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:38:30 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data7/adaptor.mars.internal-1597235906.569233-23942-4-396d46d8-8510-4c67-a995-f0a6779b8ae2.nc /cache/tmp/396d46d8-8510-4c67-a995-f0a6779b8ae2-adaptor.mars.internal-1597235906.5699115-23942-1-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2005_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:27 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2005, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:38:39 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data7/adaptor.mars.internal-1597235914.0962694-16860-4-1c757a79-b5fc-42b3-9118-91b92d9bcecf.nc /cache/tmp/1c757a79-b5fc-42b3-9118-91b92d9bcecf-adaptor.mars.internal-1597235914.0969098-16860-1-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1996_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:43:54 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 1996, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:30:38 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data5/adaptor.mars.internal-1597235433.7833984-4764-24-ce92501b-d666-4ad5-b5fc-8f56c8328f25.nc /cache/tmp/ce92501b-d666-4ad5-b5fc-8f56c8328f25-adaptor.mars.internal-1597235433.7844527-4764-8-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2003_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2010_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2003_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:05 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2003, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:41:15 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data0/adaptor.mars.internal-1597236070.6116223-32034-8-98f4d052-bae7-4f4c-9427-5aef7027c192.nc /cache/tmp/98f4d052-bae7-4f4c-9427-5aef7027c192-adaptor.mars.internal-1597236070.6121285-32034-3-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2014_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:45:00 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2014, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:44:47 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data2/adaptor.mars.internal-1597236283.9633782-13797-23-6d91ceb0-09e8-42fb-a2e1-fca76c2f8dd1.nc /cache/tmp/6d91ceb0-09e8-42fb-a2e1-fca76c2f8dd1-adaptor.mars.internal-1597236283.9644802-13797-7-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2002_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2000_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1999_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:06 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 1999, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:41:15 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data8/adaptor.mars.internal-1597236071.0705113-31958-13-60653d0f-2fb4-4044-91ec-97bebb381779.nc /cache/tmp/60653d0f-2fb4-4044-91ec-97bebb381779-adaptor.mars.internal-1597236071.0710933-31958-3-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc">
    <attribute:alias>native6</attribute:alias>
    <attribute:dataset>ERA5</attribute:dataset>
    <attribute:diagnostic>calculate_weights_climwip</attribute:diagnostic>
    <attribute:end_year>2014</attribute:end_year>
    <attribute:filename>/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc</attribute:filename>
    <attribute:frequency>mon</attribute:frequency>
    <attribute:long_name>Near-Surface Air Temperature</attribute:long_name>
    <attribute:mip>Amon</attribute:mip>
    <attribute:modeling_realm>['atmos']</attribute:modeling_realm>
    <attribute:original_short_name>tas</attribute:original_short_name>
    <attribute:preprocessor>detrended_std</attribute:preprocessor>
    <attribute:project>native6</attribute:project>
    <attribute:recipe_dataset_index>2</attribute:recipe_dataset_index>
    <attribute:short_name>tas</attribute:short_name>
    <attribute:standard_name>air_temperature</attribute:standard_name>
    <attribute:start_year>1995</attribute:start_year>
    <attribute:tier>3</attribute:tier>
    <attribute:type>reanaly</attribute:type>
    <attribute:units>K</attribute:units>
    <attribute:variable_group>tas</attribute:variable_group>
    <attribute:version>1</attribute:version>
    <preprocessor:cleanup>{'remove': ['/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014_fixed']}</preprocessor:cleanup>
    <preprocessor:climate_statistics>{'operator': 'std_dev'}</preprocessor:climate_statistics>
    <preprocessor:cmor_check_data>{'cmor_table': 'native6', 'mip': 'Amon', 'short_name': 'tas', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:cmor_check_data>
    <preprocessor:cmor_check_metadata>{'cmor_table': 'native6', 'mip': 'Amon', 'short_name': 'tas', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:cmor_check_metadata>
    <preprocessor:concatenate>{}</preprocessor:concatenate>
    <preprocessor:detrend>{'dimension': 'time', 'method': 'linear'}</preprocessor:detrend>
    <preprocessor:extract_region>{'start_longitude': -10.0, 'end_longitude': 39.0, 'start_latitude': 30.0, 'end_latitude': 76.25}</preprocessor:extract_region>
    <preprocessor:extract_time>{'start_year': 1995, 'end_year': 2015, 'start_month': 1, 'end_month': 1, 'start_day': 1, 'end_day': 1}</preprocessor:extract_time>
    <preprocessor:fix_data>{'project': 'native6', 'dataset': 'ERA5', 'short_name': 'tas', 'mip': 'Amon', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:fix_data>
    <preprocessor:fix_file>{'project': 'native6', 'dataset': 'ERA5', 'short_name': 'tas', 'mip': 'Amon', 'output_dir': '/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014_fixed'}</preprocessor:fix_file>
    <preprocessor:fix_metadata>{'project': 'native6', 'dataset': 'ERA5', 'short_name': 'tas', 'mip': 'Amon', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:fix_metadata>
    <preprocessor:load>{'callback': &lt;function concatenate_callback at 0x7f0ea9982dc0&gt;}</preprocessor:load>
    <preprocessor:mask_landsea>{'mask_out': 'sea', 'fx_variables': {'sftlf': [], 'sftof': []}}</preprocessor:mask_landsea>
    <preprocessor:regrid>{'target_grid': '2.5x2.5', 'scheme': 'linear'}</preprocessor:regrid>
    <preprocessor:save>{'compress': False, 'filename': '/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc'}</preprocessor:save>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1995_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:43:46 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 1995, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:41:15 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data5/adaptor.mars.internal-1597236070.6037593-13797-12-5448dff2-fb2a-473d-a580-03e9b77de42e.nc /cache/tmp/5448dff2-fb2a-473d-a580-03e9b77de42e-adaptor.mars.internal-1597236070.604191-13797-3-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2013_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1998_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2000_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:43:31 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2000, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:30:38 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data8/adaptor.mars.internal-1597235433.78062-5670-8-a07277c0-28cd-4761-bd42-fd65b4f17e57.nc /cache/tmp/a07277c0-28cd-4761-bd42-fd65b4f17e57-adaptor.mars.internal-1597235433.78109-5670-4-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2008_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:31 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2008, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:30:47 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data5/adaptor.mars.internal-1597235441.2776072-1151-28-77f85b4e-3903-40e2-b3a3-1b744987c3f6.nc /cache/tmp/77f85b4e-3903-40e2-b3a3-1b744987c3f6-adaptor.mars.internal-1597235441.278114-1151-6-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2007_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:13 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2007, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:41:49 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data2/adaptor.mars.internal-1597236106.8865974-8425-12-34606ba2-9a88-4d8b-963f-4dee140afe05.nc /cache/tmp/34606ba2-9a88-4d8b-963f-4dee140afe05-adaptor.mars.internal-1597236106.8871615-8425-4-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1997_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:43:30 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 1997, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:38:37 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data9/adaptor.mars.internal-1597235912.801381-32696-12-62f1781f-1662-4cfa-8ecc-61d4622af7ca.nc /cache/tmp/62f1781f-1662-4cfa-8ecc-61d4622af7ca-adaptor.mars.internal-1597235912.8020194-32696-4-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1997_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2008_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2006_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:05 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2006, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:39:54 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data0/adaptor.mars.internal-1597235991.3872252-21301-13-9b060625-e86e-4dd6-b8e0-15f5616ba1a7.nc /cache/tmp/9b060625-e86e-4dd6-b8e0-15f5616ba1a7-adaptor.mars.internal-1597235991.3877287-21301-5-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2005_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2012_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:22 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2012, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:44:06 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data6/adaptor.mars.internal-1597236243.006152-3084-30-bc411e74-da71-432d-bbce-60c2c544c1df.nc /cache/tmp/bc411e74-da71-432d-bbce-60c2c544c1df-adaptor.mars.internal-1597236243.006705-3084-5-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2013_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:40 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2013, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:44:35 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data8/adaptor.mars.internal-1597236270.8645675-22970-27-e1c6c934-1958-476b-8e56-dd7b803de794.nc /cache/tmp/e1c6c934-1958-476b-8e56-dd7b803de794-adaptor.mars.internal-1597236270.88617-22970-9-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1996_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2004_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1998_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:43:48 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 1998, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:39:54 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data8/adaptor.mars.internal-1597235989.7003796-27473-13-8ecc3994-a411-4af3-81eb-c7ebcc3f130e.nc /cache/tmp/8ecc3994-a411-4af3-81eb-c7ebcc3f130e-adaptor.mars.internal-1597235989.7103078-27473-3-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2006_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2007_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1995_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2004_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:14 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2004, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:30:47 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data7/adaptor.mars.internal-1597235441.2982569-30719-24-d76218ca-91f9-4a53-b860-4f0cee95c4fc.nc /cache/tmp/d76218ca-91f9-4a53-b860-4f0cee95c4fc-adaptor.mars.internal-1597235441.299189-30719-6-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2014_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2011_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:30 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2011, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:41:15 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data0/adaptor.mars.internal-1597236070.8609712-17270-26-462cbbbd-ffc0-4c13-bd95-c0e64caccdca.nc /cache/tmp/462cbbbd-ffc0-4c13-bd95-c0e64caccdca-adaptor.mars.internal-1597236070.862182-17270-9-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2011_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2009_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:08 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2009, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:38:41 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data9/adaptor.mars.internal-1597235915.7341604-14948-11-f872f1e9-e83c-4f44-a30b-fe7cdf14dbab.nc /cache/tmp/f872f1e9-e83c-4f44-a30b-fe7cdf14dbab-adaptor.mars.internal-1597235915.734653-14948-4-tmp.grib</attribute:history>
  </prov:entity>
  <prov:activity prov:id="task:calculate_weights_climwip/tas"/>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2010_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:44:24 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2010, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:39:54 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data6/adaptor.mars.internal-1597235991.396578-8425-5-9396b03e-f47d-4bf7-9458-ac745835f8a0.nc /cache/tmp/9396b03e-f47d-4bf7-9458-ac745835f8a0-adaptor.mars.internal-1597235991.3971033-8425-2-tmp.grib</attribute:history>
  </prov:entity>
  <prov:entity prov:id="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2002_monthly.nc">
    <attribute:Conventions>CF-1.6</attribute:Conventions>
    <attribute:history>2020-08-12 12:43:48 UTC by era5cli 1.0.0: reanalysis-era5-single-levels-monthly-means {'variable': '2m_temperature', 'year': 2002, 'product_type': 'monthly_averaged_reanalysis', 'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 'time': ['00:00'], 'format': 'netcdf'}
2020-08-12 12:39:47 GMT by grib_to_netcdf-2.16.0: /opt/ecmwf/eccodes/bin/grib_to_netcdf -S param -o /cache/data3/adaptor.mars.internal-1597235983.0009174-9919-3-3543c173-dfe2-4ab0-b42c-7af7afff5296.nc /cache/tmp/3543c173-dfe2-4ab0-b42c-7af7afff5296-adaptor.mars.internal-1597235983.0014606-9919-2-tmp.grib</attribute:history>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_2001_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/obs_inputpath/Tier3/ERA5/1/mon/tas/era5_2m_temperature_1999_monthly.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasStartedBy>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
    <prov:trigger prov:ref="recipe:recipe_climwip.yml"/>
    <prov:starter prov:ref="software:esmvaltool==2.1.0"/>
  </prov:wasStartedBy>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc"/>
    <prov:usedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/native6_ERA5_reanaly_1_Amon_tas_1995-2014.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/climwip"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/default_inputpath/tas_Amon_ACCESS1-0_historical_r1i1p1_185001-200512.nc">
    <attribute:Conventions>CF-1.4</attribute:Conventions>
    <attribute:branch_time>109207.0</attribute:branch_time>
    <attribute:cmor_version>2.8.0</attribute:cmor_version>
    <attribute:contact>The ACCESS wiki: http://wiki.csiro.au/confluence/display/ACCESS/Home. Contact Tony.Hirst@csiro.au regarding the ACCESS coupled climate model. Contact Peter.Uhe@csiro.au regarding ACCESS coupled climate model CMIP5 datasets.</attribute:contact>
    <attribute:creation_date>2012-01-18T23:37:47Z</attribute:creation_date>
    <attribute:experiment>historical</attribute:experiment>
    <attribute:experiment_id>historical</attribute:experiment_id>
    <attribute:forcing>GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)</attribute:forcing>
    <attribute:frequency>mon</attribute:frequency>
    <attribute:history>CMIP5 compliant file produced from raw ACCESS model output using the ACCESS Post-Processor and CMOR2. 2012-01-18T23:37:47Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.</attribute:history>
    <attribute:initialization_method>1</attribute:initialization_method>
    <attribute:institute_id>CSIRO-BOM</attribute:institute_id>
    <attribute:institution>CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), and BOM (Bureau of Meteorology, Australia)</attribute:institution>
    <attribute:model_id>ACCESS1-0</attribute:model_id>
    <attribute:modeling_realm>atmos</attribute:modeling_realm>
    <attribute:parent_experiment>pre-industrial control</attribute:parent_experiment>
    <attribute:parent_experiment_id>piControl</attribute:parent_experiment_id>
    <attribute:parent_experiment_rip>r1i1p1</attribute:parent_experiment_rip>
    <attribute:physics_version>1</attribute:physics_version>
    <attribute:product>output</attribute:product>
    <attribute:project_id>CMIP5</attribute:project_id>
    <attribute:realization>1</attribute:realization>
    <attribute:references>See http://wiki.csiro.au/confluence/display/ACCESS/ACCESS+Publications</attribute:references>
    <attribute:source>ACCESS1-0 2011. Atmosphere: AGCM v1.0 (N96 grid-point, 1.875 degrees EW x approx 1.25 degree NS, 38 levels); ocean: NOAA/GFDL MOM4p1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S, 50 levels); sea ice: CICE4.1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S); land: MOSES2 (1.875 degree EW x 1.25 degree NS, 4 levels</attribute:source>
    <attribute:table_id>Table Amon (27 April 2011) 9c851218e3842df9a62ef38b1e2575bb</attribute:table_id>
    <attribute:title>ACCESS1-0 model output prepared for CMIP5 historical</attribute:title>
    <attribute:tracking_id>9dada092-9a89-4283-b128-e350d82324c1</attribute:tracking_id>
    <attribute:version_number>v20120115</attribute:version_number>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/default_inputpath/tas_Amon_ACCESS1-0_rcp45_r1i1p1_200601-210012.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/default_inputpath/tas_Amon_ACCESS1-0_rcp45_r1i1p1_200601-210012.nc">
    <attribute:Conventions>CF-1.4</attribute:Conventions>
    <attribute:branch_time>732311.0</attribute:branch_time>
    <attribute:cmor_version>2.8.0</attribute:cmor_version>
    <attribute:contact>The ACCESS wiki: http://wiki.csiro.au/confluence/display/ACCESS/Home. Contact Tony.Hirst@csiro.au regarding the ACCESS coupled climate model. Contact Peter.Uhe@csiro.au regarding ACCESS coupled climate model CMIP5 datasets.</attribute:contact>
    <attribute:creation_date>2012-01-25T04:46:37Z</attribute:creation_date>
    <attribute:experiment>RCP4.5</attribute:experiment>
    <attribute:experiment_id>rcp45</attribute:experiment_id>
    <attribute:forcing>GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)</attribute:forcing>
    <attribute:frequency>mon</attribute:frequency>
    <attribute:history>CMIP5 compliant file produced from raw ACCESS model output using the ACCESS Post-Processor and CMOR2. 2012-01-25T04:46:37Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.</attribute:history>
    <attribute:initialization_method>1</attribute:initialization_method>
    <attribute:institute_id>CSIRO-BOM</attribute:institute_id>
    <attribute:institution>CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), and BOM (Bureau of Meteorology, Australia)</attribute:institution>
    <attribute:model_id>ACCESS1-0</attribute:model_id>
    <attribute:modeling_realm>atmos</attribute:modeling_realm>
    <attribute:parent_experiment>historical</attribute:parent_experiment>
    <attribute:parent_experiment_id>historical</attribute:parent_experiment_id>
    <attribute:parent_experiment_rip>r1i1p1</attribute:parent_experiment_rip>
    <attribute:physics_version>1</attribute:physics_version>
    <attribute:product>output</attribute:product>
    <attribute:project_id>CMIP5</attribute:project_id>
    <attribute:realization>1</attribute:realization>
    <attribute:references>See http://wiki.csiro.au/confluence/display/ACCESS/ACCESS+Publications</attribute:references>
    <attribute:source>ACCESS1-0 2011. Atmosphere: AGCM v1.0 (N96 grid-point, 1.875 degrees EW x approx 1.25 degree NS, 38 levels); ocean: NOAA/GFDL MOM4p1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S, 50 levels); sea ice: CICE4.1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S); land: MOSES2 (1.875 degree EW x 1.25 degree NS, 4 levels</attribute:source>
    <attribute:table_id>Table Amon (27 April 2011) 9c851218e3842df9a62ef38b1e2575bb</attribute:table_id>
    <attribute:title>ACCESS1-0 model output prepared for CMIP5 RCP4.5</attribute:title>
    <attribute:tracking_id>e533295a-be80-42e9-80f6-e8349b179486</attribute:tracking_id>
    <attribute:version_number>v20120115</attribute:version_number>
  </prov:entity>
  <prov:entity prov:id="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014.nc">
    <attribute:alias>CMIP5_historical-rcp45</attribute:alias>
    <attribute:dataset>ACCESS1-0</attribute:dataset>
    <attribute:diagnostic>calculate_weights_climwip</attribute:diagnostic>
    <attribute:end_year>2014</attribute:end_year>
    <attribute:ensemble>r1i1p1</attribute:ensemble>
    <attribute:exp>['historical', 'rcp45']</attribute:exp>
    <attribute:filename>/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014.nc</attribute:filename>
    <attribute:frequency>mon</attribute:frequency>
    <attribute:institute>['CSIRO-BOM']</attribute:institute>
    <attribute:long_name>Near-Surface Air Temperature</attribute:long_name>
    <attribute:mip>Amon</attribute:mip>
    <attribute:modeling_realm>['atmos']</attribute:modeling_realm>
    <attribute:original_short_name>tas</attribute:original_short_name>
    <attribute:preprocessor>detrended_std</attribute:preprocessor>
    <attribute:project>CMIP5</attribute:project>
    <attribute:recipe_dataset_index>0</attribute:recipe_dataset_index>
    <attribute:short_name>tas</attribute:short_name>
    <attribute:standard_name>air_temperature</attribute:standard_name>
    <attribute:start_year>1995</attribute:start_year>
    <attribute:units>K</attribute:units>
    <attribute:variable_group>tas</attribute:variable_group>
    <preprocessor:cleanup>{'remove': ['/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014_fixed']}</preprocessor:cleanup>
    <preprocessor:climate_statistics>{'operator': 'std_dev'}</preprocessor:climate_statistics>
    <preprocessor:cmor_check_data>{'cmor_table': 'CMIP5', 'mip': 'Amon', 'short_name': 'tas', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:cmor_check_data>
    <preprocessor:cmor_check_metadata>{'cmor_table': 'CMIP5', 'mip': 'Amon', 'short_name': 'tas', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:cmor_check_metadata>
    <preprocessor:concatenate>{}</preprocessor:concatenate>
    <preprocessor:detrend>{'dimension': 'time', 'method': 'linear'}</preprocessor:detrend>
    <preprocessor:extract_region>{'start_longitude': -10.0, 'end_longitude': 39.0, 'start_latitude': 30.0, 'end_latitude': 76.25}</preprocessor:extract_region>
    <preprocessor:extract_time>{'start_year': 1995, 'end_year': 2015, 'start_month': 1, 'end_month': 1, 'start_day': 1, 'end_day': 1}</preprocessor:extract_time>
    <preprocessor:fix_data>{'project': 'CMIP5', 'dataset': 'ACCESS1-0', 'short_name': 'tas', 'mip': 'Amon', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:fix_data>
    <preprocessor:fix_file>{'project': 'CMIP5', 'dataset': 'ACCESS1-0', 'short_name': 'tas', 'mip': 'Amon', 'output_dir': '/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014_fixed'}</preprocessor:fix_file>
    <preprocessor:fix_metadata>{'project': 'CMIP5', 'dataset': 'ACCESS1-0', 'short_name': 'tas', 'mip': 'Amon', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:fix_metadata>
    <preprocessor:load>{'callback': &lt;function concatenate_callback at 0x7f0ea9982dc0&gt;}</preprocessor:load>
    <preprocessor:mask_landsea>{'mask_out': 'sea', 'fx_variables': {'sftlf': [], 'sftof': []}}</preprocessor:mask_landsea>
    <preprocessor:regrid>{'target_grid': '2.5x2.5', 'scheme': 'linear'}</preprocessor:regrid>
    <preprocessor:save>{'compress': False, 'filename': '/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014.nc'}</preprocessor:save>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/default_inputpath/tas_Amon_ACCESS1-0_historical_r1i1p1_185001-200512.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc"/>
    <prov:usedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp45_r1i1p1_tas_1995-2014.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/climwip"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/default_inputpath/tas_Amon_ACCESS1-0_historical_r1i1p1_185001-200512.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:entity prov:id="file:/home/stef/default_inputpath/tas_Amon_ACCESS1-0_rcp85_r1i1p1_200601-210012.nc">
    <attribute:Conventions>CF-1.4</attribute:Conventions>
    <attribute:branch_time>732311.0</attribute:branch_time>
    <attribute:cmor_version>2.8.0</attribute:cmor_version>
    <attribute:contact>The ACCESS wiki: http://wiki.csiro.au/confluence/display/ACCESS/Home. Contact Tony.Hirst@csiro.au regarding the ACCESS coupled climate model. Contact Peter.Uhe@csiro.au regarding ACCESS coupled climate model CMIP5 datasets.</attribute:contact>
    <attribute:creation_date>2012-01-31T20:53:26Z</attribute:creation_date>
    <attribute:experiment>RCP8.5</attribute:experiment>
    <attribute:experiment_id>rcp85</attribute:experiment_id>
    <attribute:forcing>GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)</attribute:forcing>
    <attribute:frequency>mon</attribute:frequency>
    <attribute:history>CMIP5 compliant file produced from raw ACCESS model output using the ACCESS Post-Processor and CMOR2. 2012-01-31T20:53:26Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.</attribute:history>
    <attribute:initialization_method>1</attribute:initialization_method>
    <attribute:institute_id>CSIRO-BOM</attribute:institute_id>
    <attribute:institution>CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), and BOM (Bureau of Meteorology, Australia)</attribute:institution>
    <attribute:model_id>ACCESS1-0</attribute:model_id>
    <attribute:modeling_realm>atmos</attribute:modeling_realm>
    <attribute:parent_experiment>historical</attribute:parent_experiment>
    <attribute:parent_experiment_id>historical</attribute:parent_experiment_id>
    <attribute:parent_experiment_rip>r1i1p1</attribute:parent_experiment_rip>
    <attribute:physics_version>1</attribute:physics_version>
    <attribute:product>output</attribute:product>
    <attribute:project_id>CMIP5</attribute:project_id>
    <attribute:realization>1</attribute:realization>
    <attribute:references>See http://wiki.csiro.au/confluence/display/ACCESS/ACCESS+Publications</attribute:references>
    <attribute:source>ACCESS1-0 2011. Atmosphere: AGCM v1.0 (N96 grid-point, 1.875 degrees EW x approx 1.25 degree NS, 38 levels); ocean: NOAA/GFDL MOM4p1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S, 50 levels); sea ice: CICE4.1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S); land: MOSES2 (1.875 degree EW x 1.25 degree NS, 4 levels</attribute:source>
    <attribute:table_id>Table Amon (27 April 2011) 9c851218e3842df9a62ef38b1e2575bb</attribute:table_id>
    <attribute:title>ACCESS1-0 model output prepared for CMIP5 RCP8.5</attribute:title>
    <attribute:tracking_id>a011116d-66b7-4294-bd0f-cec339c4ddb4</attribute:tracking_id>
    <attribute:version_number>v20120115</attribute:version_number>
  </prov:entity>
  <prov:entity prov:id="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014.nc">
    <attribute:alias>CMIP5_historical-rcp85</attribute:alias>
    <attribute:dataset>ACCESS1-0</attribute:dataset>
    <attribute:diagnostic>calculate_weights_climwip</attribute:diagnostic>
    <attribute:end_year>2014</attribute:end_year>
    <attribute:ensemble>r1i1p1</attribute:ensemble>
    <attribute:exp>['historical', 'rcp85']</attribute:exp>
    <attribute:filename>/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014.nc</attribute:filename>
    <attribute:frequency>mon</attribute:frequency>
    <attribute:institute>['CSIRO-BOM']</attribute:institute>
    <attribute:long_name>Near-Surface Air Temperature</attribute:long_name>
    <attribute:mip>Amon</attribute:mip>
    <attribute:modeling_realm>['atmos']</attribute:modeling_realm>
    <attribute:original_short_name>tas</attribute:original_short_name>
    <attribute:preprocessor>detrended_std</attribute:preprocessor>
    <attribute:project>CMIP5</attribute:project>
    <attribute:recipe_dataset_index>1</attribute:recipe_dataset_index>
    <attribute:short_name>tas</attribute:short_name>
    <attribute:standard_name>air_temperature</attribute:standard_name>
    <attribute:start_year>1995</attribute:start_year>
    <attribute:units>K</attribute:units>
    <attribute:variable_group>tas</attribute:variable_group>
    <preprocessor:cleanup>{'remove': ['/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014_fixed']}</preprocessor:cleanup>
    <preprocessor:climate_statistics>{'operator': 'std_dev'}</preprocessor:climate_statistics>
    <preprocessor:cmor_check_data>{'cmor_table': 'CMIP5', 'mip': 'Amon', 'short_name': 'tas', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:cmor_check_data>
    <preprocessor:cmor_check_metadata>{'cmor_table': 'CMIP5', 'mip': 'Amon', 'short_name': 'tas', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:cmor_check_metadata>
    <preprocessor:concatenate>{}</preprocessor:concatenate>
    <preprocessor:detrend>{'dimension': 'time', 'method': 'linear'}</preprocessor:detrend>
    <preprocessor:extract_region>{'start_longitude': -10.0, 'end_longitude': 39.0, 'start_latitude': 30.0, 'end_latitude': 76.25}</preprocessor:extract_region>
    <preprocessor:extract_time>{'start_year': 1995, 'end_year': 2015, 'start_month': 1, 'end_month': 1, 'start_day': 1, 'end_day': 1}</preprocessor:extract_time>
    <preprocessor:fix_data>{'project': 'CMIP5', 'dataset': 'ACCESS1-0', 'short_name': 'tas', 'mip': 'Amon', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:fix_data>
    <preprocessor:fix_file>{'project': 'CMIP5', 'dataset': 'ACCESS1-0', 'short_name': 'tas', 'mip': 'Amon', 'output_dir': '/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014_fixed'}</preprocessor:fix_file>
    <preprocessor:fix_metadata>{'project': 'CMIP5', 'dataset': 'ACCESS1-0', 'short_name': 'tas', 'mip': 'Amon', 'frequency': 'mon', 'check_level': &lt;CheckLevels.DEFAULT: 3&gt;}</preprocessor:fix_metadata>
    <preprocessor:load>{'callback': &lt;function concatenate_callback at 0x7f0ea9982dc0&gt;}</preprocessor:load>
    <preprocessor:mask_landsea>{'mask_out': 'sea', 'fx_variables': {'sftlf': [], 'sftof': []}}</preprocessor:mask_landsea>
    <preprocessor:regrid>{'target_grid': '2.5x2.5', 'scheme': 'linear'}</preprocessor:regrid>
    <preprocessor:save>{'compress': False, 'filename': '/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014.nc'}</preprocessor:save>
  </prov:entity>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014.nc"/>
    <prov:usedEntity prov:ref="file:/home/stef/default_inputpath/tas_Amon_ACCESS1-0_rcp85_r1i1p1_200601-210012.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/tas"/>
  </prov:wasDerivedFrom>
  <prov:wasDerivedFrom>
    <prov:generatedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/work/calculate_weights_climwip/climwip/performance_tas.nc"/>
    <prov:usedEntity prov:ref="file:/mnt/e/eScience/climwip/esmvaltool_output/recipe_climwip_20201028_161147/preproc/calculate_weights_climwip/tas/CMIP5_ACCESS1-0_Amon_historical-rcp85_r1i1p1_tas_1995-2014.nc"/>
    <prov:activity prov:ref="task:calculate_weights_climwip/climwip"/>
  </prov:wasDerivedFrom>
</prov:document>
F    	   software                     Created with ESMValTool v2.1.0O       caption    /          Performance metric (RMS error) for variable tas+ъ3