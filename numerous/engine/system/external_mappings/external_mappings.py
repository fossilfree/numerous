import numpy as np

from numerous.engine.system.external_mappings.interpolation_type import InterpolationType


class ExternalMapping:
    def __init__(self, external_mappings):
        self.external_mappings = external_mappings
        self.external_mappings_numpy = []
        self.external_mappings_time = []
        self.external_mappings_time = []
        self.external_columns = []
        self.interpolation_type = []
        self.t_max = 0
        for external_mapping in self.external_mappings:
            for element in external_mapping.external_mappings:
                df = external_mapping.data_loader.load(element.data_frame_id, element.index_to_timestep_mapping_start)
                element.add_df(df)
        for external_mapping in self.external_mappings:
            for element in external_mapping.external_mappings:
                self.external_columns.append(list(element.dataframe_aliases.keys()))
                self.interpolation_type.append([a_tuple[1] for a_tuple in list(element.dataframe_aliases.values())])
                self.external_mappings_numpy.append(
                    element.df[[a_tuple[0] for a_tuple in list(element.dataframe_aliases.values())]]
                    .to_numpy(dtype=np.float64)[element.index_to_timestep_mapping_start:])
                self.external_mappings_time.append(
                    element.time_multiplier * element.df[element.index_to_timestep_mapping]
                    .to_numpy(dtype=np.float64)[element.index_to_timestep_mapping_start:])
                # TODO extend for multiple dataframes
                self.t_max = np.max(self.external_mappings_time[0])
                self.t_min = np.min(self.external_mappings_time[0])

        self.external_mappings_numpy = np.array(self.external_mappings_numpy, dtype=np.float64)
        self.external_mappings_time = np.array(self.external_mappings_time, dtype=np.float64)
        self.interpolation_type = [item for sublist in self.interpolation_type for item in sublist]
        self.external_df_idx = []
        self.interpolation_info = []

    def load_new_external_data_batch(self, t):
        self.external_mappings_numpy = []
        self.external_mappings_time = []

        for external_mapping in self.external_mappings:
            for element in external_mapping.external_mappings:
                # TODO division round bugs? we can skip a row here
                df = external_mapping.data_loader.load(element.data_frame_id, t)

                if df.empty:
                    return False

                element.add_df(df)
        for external_mapping in self.external_mappings:
            for element in external_mapping.external_mappings:
                self.external_mappings_numpy.append(
                    element.df[[a_tuple[0] for a_tuple in list(element.dataframe_aliases.values())]]
                    .to_numpy(dtype=np.float64)[element.index_to_timestep_mapping_start:])
                self.external_mappings_time.append(
                    element.time_multiplier * element.df[element.index_to_timestep_mapping]
                    .to_numpy(dtype=np.float64)[element.index_to_timestep_mapping_start:])

        self.external_mappings_numpy = np.array(self.external_mappings_numpy, dtype=np.float64)
        self.external_mappings_time = np.array(self.external_mappings_time, dtype=np.float64)
        # TODO extend for multiple dataframes
        self.t_max = np.max(self.external_mappings_time[0])
        self.t_min = np.min(self.external_mappings_time[0])
        return True

    def store_mappings(self):
        self.external_df_idx = np.array(self.external_df_idx, dtype=np.int64)
        self.interpolation_info = np.array(self.interpolation_info, dtype=np.bool)

    def add_df_idx(self, variables, var_id, system_id):
        for i, external_column in enumerate(self.external_columns):
            for path in variables[var_id].path.path[system_id]:
                for index, column in enumerate(external_column):
                    if column in path:
                        self.external_df_idx.append((i, index))
                        self.interpolation_info.append(
                            self.interpolation_type[index].value == InterpolationType.LINEAR.value)

    def is_mapped_var(self, variables, var_id, system_id):
        for path in variables[var_id].path.path[system_id]:
            for columns in self.external_columns:
                for column in columns:
                    if column in path:
                        return True


class ExternalMappingElement:

    def __init__(self, data_frame_id, index_to_timestep_mapping, index_to_timestep_mapping_start, time_multiplier,
                 dataframe_aliases):
        self.data_frame_id = data_frame_id
        self.index_to_timestep_mapping = index_to_timestep_mapping
        self.index_to_timestep_mapping_start = index_to_timestep_mapping_start
        self.time_multiplier = time_multiplier
        self.dataframe_aliases = dataframe_aliases
        self.df = None

    # TODO check that e_m is correct format (have all the mappings) after loading
    def add_df(self, df):
        if self.df is not None:
            columns = self.df.columns
            self.df = df
            self.df.columns = columns
        else:
            self.df = df


class ExternalMappingUnpacked:
    def __init__(self, external_mappings=None, data_loader=None):
        self.data_loader = data_loader
        self.external_mappings = external_mappings


class EmptyMapping:
    def __init__(self):
        self.external_mappings_numpy = np.empty([0, 0, 0], dtype=np.float64)
        self.external_mappings_time = np.empty([0, 0], dtype=np.float64)
        self.external_df_idx = np.empty([0, 0], dtype=np.int64)
        self.t_max = 0
        self.t_min = 0
        self.interpolation_info = np.empty([0], dtype=bool)

    def store_mappings(self):
        pass

    def is_mapped_var(self, variables, var_id, system_id):
        pass
