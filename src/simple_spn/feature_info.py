import numpy as np


class FeatureInfo(object):

    def __init__(self, value_dict):
        self.value_dict = value_dict

        self.name_dict = {}
        for f_id, [f_type, f_name, val_mapping] in value_dict.items():
            if f_type == "discrete":
                new_val_mapping = {val_name: map_val for map_val, val_name in val_mapping.items()}
            elif f_type == "numeric":
                new_val_mapping = val_mapping
            else:
                raise Exception("Unknown feature-type: " + str(f_type))
            self.name_dict[f_name] = [f_type, f_id, new_val_mapping]

    def get_num_features(self):
        return len(self.value_dict)

    def get_feature_name(self, f_id):
        return self.value_dict[f_id][1]

    def get_feature_names(self, f_ids):
        return [self.value_dict[f_id][1] for f_id in f_ids]

    def get_all_feature_names(self):
        return [f_name for _, [_, f_name, _] in self.value_dict.items()]

    def get_feature_id(self, f_name):
        return self.name_dict[f_name][1]

    def get_feature_ids(self, f_names):
        return [self.name_dict[f_name][1] for f_name in f_names]

    def get_feature_type(self, f):
        if type(f) == str:
            return self.name_dict[f][0]
        else:
            return self.value_dict[f][0]

    def all_discrete(self):
        return np.all([f_type == "discrete" for _, [f_type, _, _] in self.value_dict.items()])

    def all_numeric(self):
        return np.all([f_type == "numeric" for _, [f_type, _, _] in self.value_dict.items()])

    def is_discrete(self, f):
        if type(f) == str:
            return self.name_dict[f][0] == "discrete"
        else:
            return self.value_dict[f][0] == "discrete"

    def get_value_mapping(self, f):
        if type(f) == str:
            assert (self.name_dict[f][0] == "discrete")
            return self.value_dict[self.get_feature_id(f)][2]
        else:
            assert (self.value_dict[f][0] == "discrete")
            return self.value_dict[f][2]

    def get_inverse_value_mapping(self, f):
        if type(f) == str:
            assert (self.name_dict[f][0] == "discrete")
            return self.name_dict[f][2]
        else:
            assert (self.value_dict[f][0] == "discrete")
            return self.name_dict[self.get_feature_name(f)][2]

    def get_num_values(self, f):
        if type(f) == str:
            assert (self.name_dict[f][0] == "discrete")
            return len(self.name_dict[f][2])
        else:
            assert (self.value_dict[f][0] == "discrete")
            return len(self.value_dict[f][2])

    def is_numeric(self, f):
        if type(f) == str:
            return self.name_dict[f][0] == "numeric"
        else:
            return self.value_dict[f][0] == "numeric"

    def get_min_value(self, f):
        if type(f) == str:
            assert (self.name_dict[f][0] == "numeric")
            return self.value_dict[self.get_feature_id(f)][2][0]
        else:
            assert (self.value_dict[f][0] == "numeric")
            return self.value_dict[f][2][0]

    def get_max_value(self, f):
        if type(f) == str:
            assert (self.name_dict[f][0] == "numeric")
            return self.value_dict[self.get_feature_id(f)][2][1]
        else:
            assert (self.value_dict[f][0] == "numeric")
            return self.value_dict[f][2][1]

    def get_value(self, f, val):
        if type(f) == str:
            if self.name_dict[f][0] == "discrete":
                return self.name_dict[f][2][val]
            else:
                return val
        else:
            if self.value_dict[f][0] == "discrete":
                return self.name_dict[self.get_feature_id(f)][2][val]
            else:
                return val

    def get_value_name(self, f, val):
        if type(f) == str:
            if self.name_dict[f][0] == "discrete":
                return self.value_dict[self.get_feature_id(f)][2][val]
            else:
                return val
        else:
            if self.value_dict[f][0] == "discrete":
                return self.value_dict[f][2][val]
            else:
                return val
