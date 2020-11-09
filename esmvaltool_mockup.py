from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint, pformat
from tqdm.auto import tqdm
from datetime import datetime
import random
import time
import yaml


recipes = (
    '...',
    'recipe_carvalhais14nat.yml',
    'recipe_climwip.yml',
    'recipe_clouds_bias.yml',
    'recipe_clouds_ipcc.yml',
    'recipe_collins13ipcc.yml',
    'recipe_combined_indices.yml',
    'recipe_consecdrydays.yml',
    'recipe_cox18nature.yml',
    'recipe_cvdp.yml',
    'examples/recipe_python_object_oriented.yml',
    'examples/recipe_preprocessor_test.yml',
    'examples/recipe_correlation.yml',
    'examples/recipe_python.yml',
    'cmorizers/recipe_era5.yml',
    'hydrology/recipe_marrmot.yml',
    'hydrology/recipe_hype.yml',
    '...',
)


_recipe = """
# ESMValTool
# recipe_climwip.yml
---
documentation:
  description: EUCP ClimWIP

  authors:
    - kalverla_peter
    - smeets_stef
    - brunner_lukas
    - camphuijsen_jaro

  maintainer:
    - kalverla_peter
    - smeets_stef
    - brunner_lukas

  references:
    - brunner2019
    - lorenz2018
    - knutti2017

datasets: &model_data
  - {dataset: ACCESS1-0, project: CMIP5, mip: Amon, exp: [historical, rcp45], ensemble: r1i1p1}
  - {dataset: ACCESS1-0, project: CMIP5, mip: Amon, exp: [historical, rcp85], ensemble: r1i1p1}

obs_data: &obs_data  # for climwip performance metrics
  - {dataset: ERA5, project: native6, type: reanaly, version: '1', tier: 3}

preprocessors:
  climwip_general: &general
    regrid:
      target_grid: 2.5x2.5
      scheme: linear
    mask_landsea:
      mask_out: sea
    extract_region:
      start_longitude: -10.0
      end_longitude: 39.0
      start_latitude: 30.0
      end_latitude: 76.25

  climatological_mean:
    <<: *general
    climate_statistics:
      operator: mean

  detrended_std:
    <<: *general
    detrend:
      dimension: time
      method: linear
    climate_statistics:
      operator: std_dev

  temperature_anomalies:
    custom_order: true
    area_statistics:
      operator: mean
    annual_statistics:
      operator: mean
    anomalies:
      period: full
      reference:
        start_year: 1981
        start_month: 1
        start_day: 1
        end_year: 2010
        end_month: 12
        end_day: 31
      standardize: false

diagnostics:
  calculate_weights_climwip:
    variables:
      tas:
        start_year: 1995
        end_year: 2014
        preprocessor: detrended_std
        additional_datasets: *obs_data
      pr:
        start_year: 1995
        end_year: 2014
        preprocessor: climatological_mean
        additional_datasets: *obs_data

    scripts:
      climwip:
        script: weighting/climwip.py
        shape_params:
          tas:
            sigma_d: 0.588
            sigma_s: 0.704
          pr:
            sigma_d: 0.658
            sigma_s: 0.704

  weighted_temperature_graph:
    variables:
      tas:
        start_year: 1960
        end_year: 2100
        preprocessor: temperature_anomalies
    scripts:
      weighted_temperature_graph:
        script: weighting/weighted_temperature_graph.py
        ancestors: [calculate_weights_climwip/climwip, tas]
        weights: 'weights.yml'

"""




class DatasetReference(object):
    """docstring for Dataset"""
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}({pformat(self.dataset)})'

    def get_cubes(self):
        return 'Cubes'

    def is_available(self):
        return True

    def locate(self):
        pre = '~/data/CMIP5/'
        short_name = self.dataset['short_name']
        dataset = self.dataset['dataset']
        project = self.dataset['project']
        mip = self.dataset['mip']
        exp = self.dataset['exp']
        if isinstance(exp, list):
            exp = '-'.join(exp)
        ensemble = self.dataset['ensemble']
        start_year = self.dataset['start_year']
        end_year = self.dataset['end_year']

        return f'{pre}{short_name}_{dataset}_{project}_{mip}_{exp}_{ensemble}_{start_year}-{end_year}.nc'


class DatasetList_x(object):
    """docstring for Dataset"""
    def __init__(self, *args):
        super().__init__()
        self.datasets = []
        for dataset in args:
            self.add(dataset)

    def __repr__(self):
        return f'{self.__class__.__name__}({pformat(self.datasets)})'

    def __iter__(self):
        for item in self.datasets:
            yield item

    def __len__(self):
        return len(self.datasets)

    def add(self, other):
        if isinstance(other, DatasetList):
            self.datasets.extend(other.datasets)
            return

        if not isinstance(other, DatasetReference):
            other = DatasetReference(**other)
        self.datasets.append(other)

    def get_cubes(self):
        return 'Cubes'

    def locate(self):
        return [dataset.locate() for dataset in self.datasets]

    def is_available(self):
        ret = {}
        for row in self.datasets:
            location = row.locate()
            available = row.is_available()
            ret[location] = available
        return ret



class Recipe(object):
    """docstring for ClassName"""
    def __init__(self, name='new_recipe'):
        super().__init__()

        self.name = name
        self.documentation = {}
        self.preprocessors = {}
        self.datasets = []
        self.diagnostics = {}

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r})'

    @property
    def description(self):
        print(self.documentation['description'])

    @property
    def authors(self):
        authors = self.documentation['authors']

        for author in (
        {
            "name": "Kalverla, Peter",
            "affiliation": "NLeSC, Netherlands",
            "orcid": "0000-0002-5025-7862",
        },
        {
            "name": "Smeets, Stef",
            "affiliation": "NLeSC, Netherlands",
            "orcid": "0000-0002-5413-9038",
        },
        {
            "name": "Brunner, Lukas",
            "affiliation": "ETH Zurich, Switzerland",
            "orcid": "0000-0001-5760-4524",
        },
        {
            "name": "Camphuijsen, Jaro",
            "affiliation": "NLeSC, Netherlands",
            "orcid": "0000-0002-8928-7831",
        }):
            print('- {name} ({affiliation})'.format(**author))
            print('  https://orcid.org/{orcid}'.format(**author))



    def set_description(self, description):
        self.documentation['description'] = description

    def set_authors(self, *authors):
        self.documentation['authors'] = list(authors)

    def set_maintainer(self, *maintainer):
        self.documentation['maintainer'] = list(maintainer)
    
    def set_references(self, *references):
        self.documentation['references'] = list(references)

    def add_diagnostic(self, name, diagnostic):
        self.diagnostics[name] = diagnostic

    def show(self):
        print(yaml.dump(self.recipe))

    def to_dict(self):
        return self.recipe

    def save(self, filename):
        print(f'Recipe saved to {filename!r}')

    @property
    def common_dataset_keys(self):
        return self.datasets

    def run(self, name=None, session=None):
        if name:
            to_run = [self.diagnostics[name]]
        else:
            to_run = self.diagnostics.values()

        if not session:
            session = 'recipe'

        session_dir = get_session_dir(session)

        total = len(to_run)
        with tqdm(total=total) as pbar:
            for diagnostic in to_run:
                print(f'\nStarting {diagnostic}')
                cubes = diagnostic.run(session_dir=session_dir)
                time.sleep(0.1 + random.random()/4)
                pbar.update()

    @property
    def output_plots(self):
        variables = (
            'independence_pr',
            'independence_tas',
            'performance_tas',
            'performance_pr',
            'weights_pr',
            'weights_tas',
        )
        plots = {}

        script = 'calculate_weights_climwip/climwip'

        for variable in variables:
            a, b = variable.split('_')
            description = f'{a.capitalize()} for {b}.'

            row = {
               'description': description,
               'directory': Path(f'plots/{script}'),
               'filename': f'{variable}.png',
               'citation': f'{variable}_citation.bibtex',
               'provenance': f'{variable}_provenance.xml',
            }              
            plots[variable] = row

        variable = 'temperature_anomaly_graph'
        plots[variable] = {
               'description': 'Temperature anomaly graph.',
               'directory': Path('plots/weighted_temperature_graph/weighted_temperature_graph'),
               'filename': f'{variable}.png',
               'citation': f'{variable}_citation.bibtex',
               'provenance': f'{variable}_provenance.xml',
            }
        return plots

    @property
    def output_data(self):
        variables = (
            'independence_pr',
            'independence_tas',
            'performance_tas',
            'performance_pr',
            'weights_pr',
            'weights_tas',
        )
        work = {}

        script = 'calculate_weights_climwip/climwip'

        for variable in variables:
            a, b = variable.split('_')
            description = f'{a.capitalize()} for {b}.'

            row = {
               'description': description,
               'directory': Path(f'work/{script}'),
               'filename': f'{variable}.nc',
               'citation': f'{variable}_citation.bibtex',
               'provenance': f'{variable}_provenance.xml',
            }              
            work[variable] = row

        variable = 'temperature_anomalies'
        work[variable] = {
               'description': 'Temperature anomaly data.',
               'directory': Path('work/weighted_temperature_graph/weighted_temperature_graph'),
               'filename': f'{variable}.nc',
               'citation': f'{variable}_citation.bibtex',
               'provenance': f'{variable}_provenance.xml',
            }
        return work

    def add_preprocessor(self, preprocessor):
        self.preprocessors[preprocessor.name] = preprocessor


def load_recipe(filename):
    recipe = Recipe(filename)

    recipe.filename = filename
    data = recipe.recipe = yaml.full_load(_recipe)

    recipe.datasets = data['datasets']
    recipe.documentation = data['documentation']
    recipe.preprocessors = generate_preprocessors(data['preprocessors'])

    datasets = data['datasets']

    for diagnostic_name, diagnostic_task in data['diagnostics'].items():
        scripts = diagnostic_task['scripts']
        variables = diagnostic_task['variables']

        ancestors = []

        for var_name, var_group in variables.items():
            var_keys = deepcopy(variables[var_name])
            additional = var_keys.pop('additional_datasets', [])

            preproc_name = var_keys.pop('preprocessor')
            preproc = recipe.preprocessors[preproc_name]

            ds_var = [DatasetReference(**var_keys, **dataset) for dataset in datasets + additional]

            var = DatasetList(
                *ds_var,
                name=var_name,
                preprocessor=preproc,
            )

            ancestors.append(var)

        for script_name, script_dict in scripts.items():
            # What to do with this?
            existing_ancestors = script_dict.pop('ancestors', None)
            
            diag = Diagnostic(
                name=script_name,
                ancestors=ancestors,
                **script_dict,
            )

        recipe.diagnostics[diagnostic_name] = diag

    return recipe


class Preprocessor(object):
    def __init__(self, name, steps):
        self.name = name
        self.steps = {}
        self.custom_order = True

        for item in steps:
            if item == 'custom_order':
                self.custom_order = steps[item]
                continue
            elif isinstance(item, Preprocessor):
                preproc = item
                for name, step in preproc.steps.items():
                    self.steps[name] = step
                continue
            elif isinstance(item, str):
                name = item
                step = steps[item]
                if isinstance(step, dict):
                    step = PreprocessorStep(name, **step)
            else:
                assert isinstance(item, PreprocessorStep)
                step = item
                name = step.function
            
            self.steps[name] = step

    def __repr__(self):
        pre = '\n- '
        steps = pre.join([f'{step}' for step in self.steps])
        return f'{self}{pre}{steps}'

    def __str__(self):
        return f'{self.__class__.__name__}({self.name!r})'

    @property
    def order(self):
        return list(self.steps.keys())

    @classmethod
    def from_yaml(cls, yaml_code):
        dct = yaml.full_load(yaml_code)
        return cls.from_dict(dct)

    @classmethod
    def from_dict(cls, mapping):
        ret = []
        for name, values in mapping.items():
            preproc = Preprocessor(name, steps=values)
            ret.append(preproc)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


class PreprocessorStep(object):
    def __init__(self, function, **params):
        self.params = params
        self.function = function
    
    def __repr__(self):
        params = ', '.join(f'{key}={value}' for key, value in self.params.items())
        return f'{self.__class__.__name__}({self.function!r}, {params})'

    def __str__(self):
        return f'{self.__class__.__name__}({self.function!r})'

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def plot(self):
        if self.function == 'extract_region':
            import matplotlib.pyplot as plt
            from matplotlib.image import imread
            if int(self.params['start_longitude']) == 3:
                im = imread('netherlands.png')
            else:
                im = imread('europe.png')
            plt.imshow(im)
            plt.show()


def generate_preprocessors(preprocessors):
    ret = {}
    for name, steps in preprocessors.items():
        preproc = Preprocessor(name, steps)
        ret[name] = preproc

    return ret


load_data = PreprocessorStep('load_data')
cmor_check = PreprocessorStep('cmor_check')


class DatasetList(object):
    """docstring for DatasetList"""
    def __init__(self, *datasets, preprocessor=None, name='NewDatasetList'):
        super().__init__()
        self.name = name
        self.datasets = datasets
        self.preprocessor = preprocessor

        self.steps_done = []
        self.steps_queue = [load_data, cmor_check]
        self.cubes = None

        if self.preprocessor:
            self.steps_queue.append(self.preprocessor)

    def __repr__(self):
        pre = '\n '
        queue = f',{pre}'.join(f'{item}' for item in self.steps_queue)
        return f'{self.__class__.__name__}({self.name!r},\nqueue=({queue}),\ndatasets={self.datasets})'

    def regrid(self, *args, **kwargs):
        return self.add_step('regrid', *args, **kwargs)

    def mask_landsea(self, *args, **kwargs):
        return self.add_step('mask_landsea', *args, **kwargs)

    def extract_region(self, *args, **kwargs):
        return self.add_step('extract_region', *args, **kwargs)

    def add_step(self, function, *args, in_place=True, **kwargs):
        if isinstance(function, str):
            function = PreprocessorStep(function, *args, **kwargs)

        if in_place:
            self.steps_queue.append(function)
        else:
            new = deepcopy(self)
            new.steps_queue.append(function)
            return new

    def print_queue(self):
        print('Steps queue')
        print('-----------')
        for step in self.steps_done:
            print(f'Done -> {step}')
        for step in self.steps_queue:
            print(f'Todo -> {step}')
        print()

    def get_cubes(self):
        if self.cubes and not self.steps_queue:
            return self.cubes

        self.compute()

        cubes = 'Cubes'
        self.cubes = cubes  # cache result
        return cubes

    def compute(self):
        while self.steps_queue:
            step = self.steps_queue.pop(0)
            
            print(f'Running {step}')
            
            if isinstance(step, Preprocessor):
                for substep in step.steps.values():
                    time.sleep(0.1 + random.random())
                    print(f' -> {substep}')
            else:
                time.sleep(0.1 + random.random())

            self.steps_done.append(step)


def get_session_dir(session):
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = f'{session}-{now}'
    print(f'Session directory: {session_dir}')
    return session_dir


class Diagnostic(object):
    def __init__(self, script, ancestors, **kwargs):
        self.script = script
        self.ancestors = ancestors
        self.kwargs = kwargs
        self.name = Path(script).stem
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.script!r})'

    def run(self, session=None, session_dir=None):
        if not session:
            session = self.name
        
        if not session_dir:
            session_dir = get_session_dir(session)

        for ancestor in self.ancestors:
            print(f'\nWorking on {ancestor.name!r}')
            cubes = ancestor.get_cubes()
        
        print(f'\nRunning script {self.script!r}')
