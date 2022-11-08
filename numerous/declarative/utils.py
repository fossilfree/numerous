def recursive_get_attr(obj, attr_list):

    attr_ = getattr(obj, attr_list[0])

    if len(attr_list)>1:
        return recursive_get_attr(attr_, attr_list[1:])
    else:
        return attr_