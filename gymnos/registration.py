import pydoc
import importlib


class ComponentSpec:
    """
    Component spec with name and entry_point

    Parameters
    -----------
    name: str
        Component name / id
    entry_point: str
        Component path, e.g gymnos.datasets.dogs_vs_cats_cnn.DogsVsCatsCNN
    """

    def __init__(self, name, entry_point):
        self.name = name
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
        return "Spec<name={}, entry_point={}>".format(self.name, self.entry_point)


class ComponentRegistry:
    """
    Component registration

    Parameters
    -----------
    component_name: str, optional
        Component name to show on error description when exception raises

    """

    def __init__(self, component_name="component"):
        self.component_specs = {}
        self.component_name = component_name

    def register(self, name, entry_point):
        """
        Register component

        Parameters
        ----------
        name: str
            Component name to register. It must be unique
        entry_point: str
            Component path, e.g gymnos.datasets.dogs_vs_cats_cnn.DogsVsCatsCNN

        """
        if name in self.component_specs:
            raise ValueError("Cannot re-register {} with name: {}".format(self.component_name, name))

        self.component_specs[name] = ComponentSpec(name, entry_point)

    def load(self, name, **kwargs):
        """
        Load registered component

        Parameters
        ----------
        name: str
            Component name / id
        **kwargs: any
            Constructor arguments for component

        Returns
        ---------
        component
            Registered component instance
        """
        try:
            mod_name, name = name.split(":", 1)
            importlib.import_module(mod_name)  # import it so we can register external components
        except ImportError:
            raise ImportError(("A module ({}) was specified but was not found, make sure the package is installed " +
                               "with `pip install` before loading {} with name {}").format(mod_name,
                                                                                           self.component_name, name))
        except ValueError:
            pass  # we don't need to import any module

        try:
            component_spec = self.component_specs[name]
        except KeyError:
            raise ValueError("No registered {} with name: {}".format(self.component_name, name))

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

    def __contains__(self, name):
        """
        Check if registry contains the component
        """
        return name in self.component_specs
