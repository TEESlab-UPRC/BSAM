"""
This module acts as library for handling I/O actions
"""

import pandas
import numpy
import pickle
import lzma
import os
import gzip
import shutil
import contextlib

#--------------------load--------------------#
def load_dataframe_from_csv(path, index_col=0, header='infer'):
    return pandas.read_csv(path, header=header, index_col=index_col)

def load_dataframe_from_excel(path, sheet, header=9, index=[0]):
    return pandas.read_excel(path, sheetname=sheet, header=header ,index_col=index)

def load_object(saved_path, zipped=False):
    if zipped in ['gzip','xz']:
        return load_zipped_pickle(saved_path,zipped)
    elif not zipped:
        return pickle.load(open(saved_path,"rb"))
    else:
        print ('zip specification not supported')
        print ('Starting debugger')
        import ipdb;ipdb.set_trace()

def load_numpy_array(array_path):
    array = False
    if os.path.isfile(array_path):
        array = numpy.load(array_path)
    return array

def load_zipped_pickle(filename,zip_type):
    filename +='.'+zip_type
    if zip_type == 'gzip':
        zipper = gzip
    elif zip_type == 'xz':
        zipper = lzma
    else:
        print ('zip specification not supported')
        print ('Starting debugger')
        import ipdb;ipdb.set_trace()

    with zipper.open(filename, 'rb') as f:
        loaded_object = pickle.loads(f.read())
    return loaded_object

#--------------------save--------------------#
def save_dataframe_to_csv(dataframe, path, index_label, index=True):
    dataframe.to_csv(path, index_label=index_label, index=index)

def save_dataframe_to_excel(dataframe, path, sheet, startrow=9):
    dataframe.to_excel(path, sheet, index_label='label', merge_cells=False, startrow=startrow)

def save_object(data,save_path='data/results/results_pickled.dat',zipped=False):
    if zipped in ['gzip','xz']:
        save_zipped_pickle(data,save_path,zipped)
    elif not zipped:
        pickle.dump(data,open(save_path, "wb"))
    else:
        print ('zip specification not supported')
        print ('Starting debugger')
        import ipdb;ipdb.set_trace()

def save_all_agents_policies(learner,single_plant_agents):
    for agent_index, agent in enumerate(single_plant_agents):
        # hydro agents do not use any learner - the water price is set by the sysop
        if agent.plant.kind != 'hydro':
                agent.save_learner_policy()
                non_hydro_index = agent_index
    # also save the samples count - this is the same across all agents so its saved once
    # this must a non-hydro plant to make sense
    single_plant_agents[non_hydro_index].save_sample_count()

def save_numpy_array(array_path, array):
    numpy.save(array_path, array, allow_pickle=False)

def save_zipped_pickle(obj,filename,zip_type):
    filename +='.'+zip_type

    if zip_type == 'gzip':
        zipper = gzip
    elif zip_type == 'xz':
        zipper = lzma
    else:
        print ('zip specification not supported')
        print ('Starting debugger')
        import ipdb;ipdb.set_trace()

    # since it is possible for this to take a bit,
    # it is best to save under a different name & overwrite afterwards
    filename_tmp =filename+'.tmp'
    with zipper.open(filename_tmp, 'wb') as f:
        f.write(pickle.dumps(obj))

    shutil.move(filename_tmp,filename)

#--------------------delete--------------------#
def delete_files_in_folder(path):
    filenames = [f for f in os.listdir(path)]
    with contextlib.suppress(FileNotFoundError):
        for f in filenames:
            os.remove(path+f)

def delete_file(filepath,zipped=False):
    """
    Deletes the last saved temp. Meant to run at prgr end to save space
    """
    with contextlib.suppress(FileNotFoundError):
        if not zipped:
            os.remove(filepath)
        else:
            os.remove(filepath+'.'+zipped)

# filepaths
def join_path_with_root_folder(section,key,config):
    """
    Meant to be used with the config.walk function of the configobj module
    To help fix the paths by joining with the root path
    """
    # do that only in keys that end with _path
    if key != 'root_data_path' and key.split('_')[-1] == 'path':
        section[key] = config['system_datapaths']['root_data_path'] + section[key]

def create_folder(folder_path):
    """
    Makes sure that a path to a folder exists. If not it creates it.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
