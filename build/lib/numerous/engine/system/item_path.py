class ItemPath:
    """
           Represent a unique item path to item from the top of gierarchy..

          Attributes
          ----------
          path :  string
            path to item in string form
          delimiter :  string
            delimiter between item objects

    """
    def __init__(self, path, delimiter="."):
        self.path = path
        self.delimiter = delimiter
        if path:
            self.path_parsed = path.split(delimiter)
        else:
            self.path_parsed = None

    def get_top_item(self):
        """
        Returns
        -------
        Item : 'Item'
                a name of item on top of ItemPath
        """
        return self.path_parsed[0]

    def get_next_item_path(self):
        """
        Returns
        -------
        item_path : 'ItemPath'
                path to a nested items from the  item.
        """
        next_part = self.path_parsed[1:]
        if next_part:
            return ItemPath(self.delimiter.join(next_part))

    def __str__(self):
        return self.path
