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

    def __init__(self, tag=None, id=None):

        if tag:
            self.tag = tag
        if id is None:
            self.id = str(uuid.uuid1())
        else:
            if self.id is None:
                self.id = uuid.UUID(id)
        super(Node, self).__init__()

    @property
    def get_tag(self):
        """
        Returns
        -------
        tag : string
            Tag of the element.
        """

        return self.tag

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
