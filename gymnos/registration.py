import pydoc
import importlib


class ComponentSpec:
    """
    Component spec with type and entry_point

    Parameters
    -----------
    type: str
        Component type / id
    entry_point: str
        Component path, e.g gymnos.datasets.dogs_vs_cats_cnn.DogsVsCatsCNN
    """

    def __init__(self, type, entry_point):
        self.type = type
        self.entry_point = entry_point

    def load(self, **kwargs):
        """
        Load component

        Parameters
        -----------
        **kwargs: any
            Constructor parameters

        Returns
        --------
        Instance from entry point with kwargs as constructor arguments
        """
        cls = pydoc.locate(self.entry_point)
        return cls(**kwargs)

    def __repr__(self):
        return "Spec<type={}, entry_point={}>".format(self.type, self.entry_point)


class ComponentRegistry:
    """
    Component registration

    Parameters
    -----------
    component_type: str, optional
        Component type to show on error description when exception raises

    """

    def __init__(self, component_type="component"):
        self.component_specs = {}
        self.component_type = component_type

    def register(self, type, entry_point):
        """
        Register component

        Parameters
        ----------
        type: str
            Component type to register. It must be unique
        entry_point: str
            Component path, e.g gymnos.datasets.dogs_vs_cats_cnn.DogsVsCatsCNN

        """
        if type in self.component_specs:
            raise ValueError("Cannot re-register {} with type: {}".format(self.component_type, type))

        self.component_specs[type] = ComponentSpec(type, entry_point)

    def load(self, type, **kwargs):
        """
        Load registered component

        Parameters
        ----------
        type: str
            Component type / id
        **kwargs: any
            Constructor arguments for component

        Returns
        ---------
        component
            Registered component instance
        """
        try:
            mod_type, type = type.split(":", 1)
            importlib.import_module(mod_type)  # import it so we can register external components
        except ImportError:
            raise ImportError(("A module ({}) was specified but was not found, make sure the package is installed " +
                               "with `pip install` before loading {} with type {}").format(mod_type,
                                                                                           self.component_type, type))
        except ValueError:
            pass  # we don't need to import any module

        try:
            component_spec = self.component_specs[type]
        except KeyError:
            raise ValueError("No registered {} with type: {}".format(self.component_type, type))

        return component_spec.load(**kwargs)

    def all(self):
        """
        Gets all component specifications

        Returns
        -------
        list
            List with all component specifications
        """
        return self.component_specs.values()

    def __contains__(self, type):
        """
        Check if registry contains the component
        """
        return type in self.component_specs
