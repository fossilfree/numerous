import numpy as np

from model.external_mappings.approximation_type import ApproximationType


class ExternalMapping:
    def __init__(self, external_mappings):
        self.external_mappings = external_mappings
        self.external_mappings_numpy = []
        self.external_mappings_time = []
        self.external_columns = []
        self.approximation_type = []
        for (df, index_to_timestep_mapping, index_to_timestep_mapping_start,
             dataframe_aliases) in self.external_mappings:
            self.external_columns.append(list(dataframe_aliases.keys()))
            self.approximation_type.append([a_tuple[1] for a_tuple in list(dataframe_aliases.values())])
            self.external_mappings_numpy.append(df[[a_tuple[0] for a_tuple in list(dataframe_aliases.values())]
                                                ].to_numpy(dtype=np.float64))
            self.external_mappings_time.append(df[index_to_timestep_mapping].to_numpy(dtype=np.float64))
        self.external_mappings_numpy = np.array(self.external_mappings_numpy, dtype=np.float64)
        self.external_mappings_time = np.array(self.external_mappings_time, dtype=np.float64)
        self.approximation_type = [item for sublist in self.approximation_type for item in sublist]
        self.external_df_idx = []
        self.approximation_info = []

    def store_mappings(self):
        self.external_df_idx = np.array(self.external_df_idx, dtype=np.int64)
        self.approximation_info = np.array(self.approximation_info, dtype=np.bool)

    def add_df_idx(self, variables, var_id, system_id):
        for i, external_column in enumerate(self.external_columns):
            for path in variables[var_id].path.path[system_id]:
                if path in external_column:
                    i1 = external_column.index(path)
                    self.external_df_idx.append((i, i1))
                    self.approximation_info.append(
                        self.approximation_type[i].value == ApproximationType.LINEAR.value)


class EmptyMapping:
    def __init__(self):
        self.external_mappings_numpy = np.empty([0, 0, 0], dtype=np.float64)
        self.external_mappings_time = np.empty([0, 0], dtype=np.float64)
        self.external_df_idx = np.empty([0, 0], dtype=int)
        self.approximation_info = np.empty([0], dtype=np.bool)

    def store_mappings(self):
        pass
