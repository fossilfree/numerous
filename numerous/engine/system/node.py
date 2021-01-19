import uuid


class Node:
    """
        Top element in an items hierarchy. Contains fields to represent as
        uniques modelling object.


       Attributes
       ----------
       tag :  string
            Not unique tag that will be used in reports or printed output.
       id : string
            UUID assigned to the node. Generated if not provided.

    """

    def __init__(self, tag=None, id_=None):
        if not isinstance(tag, str):
            raise TypeError(f'Tag must be a string. Cannot be {type(tag)}')
        if tag:
            self.tag = tag
        else:
            self.tag = 'not_tagged'
        if id_ is None:
            self.id = self.tag + str(uuid.uuid1())
        else:
            if self.id is None:
                self.id = self.tag + uuid.UUID(id_)
        super(Node, self).__init__()


    @property
    def get_id(self):
        """
        Returns
        -------
        id : string
            UUID of the element.
        """
        return self.id

    def __repr__(self):
        return self.tag
