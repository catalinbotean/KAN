import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union

# Mapping of convolution classes to their compatible normalization classes grouped by convolution layer type
CONVOLUTION_COMPATIBILITY_MAP = {
    1: {
        "convolution": nn.Conv1d,
        "normalization": [nn.InstanceNorm1d, nn.BatchNorm1d],
        "dropout": nn.Dropout1d,
    },
    2: {
        "convolution": nn.Conv2d,
        "normalization": [nn.InstanceNorm2d, nn.BatchNorm2d],
        "dropout": nn.Dropout2d,
    },
    3: {
        "convolution": nn.Conv3d,
        "normalization": [nn.InstanceNorm3d, nn.BatchNorm3d],
        "dropout": nn.Dropout3d,
    }
}


class KANConvNDLayer(nn.Module):
    """
    Kolmogorov-Arnold Network convolutional layer with support for ND (1D, 2D, 3D) convolutions and spline-based
    transformations. This class defines a custom convolutional layer where each group applies a base linear
    transformation, spline-based transformation, and layer normalization.

    Parameters:
    -----------
    convolution_dim : int, optional, default=2
        Dimensionality of the convolution (1 for 1D, 2 for 2D, 3 for 3D).
    normalization_class : nn.Module
        Normalization layer class, e.g., nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d.
    input_dim : int
        Number of input channels.
    output_dim : int
        Number of output channels.
    spline_order : int
        Order of the spline transformation.
    kernel_size : int or tuple
        Size of the convolutional kernel.
    groups : int, optional, default=1
        Number of groups to split the input channels.
    padding : int or tuple, optional, default=0
        Padding for convolutional layers.
    stride : int or tuple, optional, default=1
        Stride for convolutional layers.
    dilation_rate : int or tuple, optional, default=1
        Dilation for convolutional layers.
    grid_size : int, optional, default=5
        Size of the grid for spline interpolation.
    base_activation_function : nn.Module, optional, default=nn.GELU
        Activation function applied to the base transformation.
    grid_range : list, optional, default=[-1, 1]
        Range of values for the spline interpolation grid.
    dropout_probability : float, optional, default=0.0
        Dropout probability.
    normalization_kwargs : dict
        Additional arguments for the normalization layer.
    """
    def __init__(self,
                 convolution_dim: int,
                 normalization_class: nn.Module,
                 input_dim: int,
                 output_dim: int,
                 spline_order: int,
                 kernel_size: Union[int, List[int]],
                 groups: int = 1,
                 padding: int = 0,
                 stride: int = 1,
                 dilation_rate: int = 1,
                 grid_size: int = 5,
                 base_activation_function: nn.Module = nn.GELU,
                 grid_range: List[float] = [-1, 1],
                 dropout_probability: float = 0.0,
                 **normalization_kwargs: Any):

        super(KANConvNDLayer, self).__init__()

        # Store configuration parameters
        self.convolution_dim = convolution_dim
        self.normalization_class = normalization_class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.grid_size = grid_size
        self.base_activation_function = base_activation_function()
        self.grid_range = grid_range
        self.dropout_probability = dropout_probability
        self.normalization_kwargs = normalization_kwargs

        # Validate the parameters
        self._validate_convolutional_layer_parameters()

        # Configure layers
        self.dropout_layer = self._configure_dropout()
        self.base_convolutional_layers = self._configure_base_convolutional_layers()
        self.spline_convolutional_layers = self._configure_spline_convolutional_layers()
        self.normalization_layers = self._configure_normalization_layers()
        self.prelu_activations = self._configure_prelu_activation_functions()

        # Create spline grid
        self.spline_grid = self._create_spline_grid(self.grid_size, self.spline_order)

        # Initialize weights
        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass, applying the transformations for each group in sequence.

        Args:
        ----
        x : torch.Tensor
            Input tensor with shape (batch, input_dim, *spatial_dims).

        Returns:
        -------
        torch.Tensor
            Output tensor after processing through all groups.
        """
        group_inputs = torch.split(x, self.input_dim // self.groups, dim=1)
        group_outputs = []
        for group_index, group_x in enumerate(group_inputs):
            group_output = self._forward_spline(group_x, group_index)
            group_outputs.append(group_output)

        return torch.cat(group_outputs, dim=1)


    def _validate_convolutional_layer_parameters(self):
        """
        Validates parameters for the KANConvNDLayer, ensuring compatibility of convolution and normalization types
        as well as valid group configuration.
        """
        self._validate_convolutional_layer_support()
        self._validate_convolutional_normalization_layers_compatibility()
        self._validate_grouping_configuration()


    def _validate_convolutional_layer_support(self):
        """
        Validates the support of the convolutional layer for the specified dimensionality.
        """
        if CONVOLUTION_COMPATIBILITY_MAP.get(self.convolution_dim)["convolution"] is None:
            raise ValueError(
                f"Unsupported convolution class: {self.convolutional_class}. Supported classes are {list(CONVOLUTION_COMPATIBILITY_MAP.keys())}.")


    def _validate_convolutional_normalization_layers_compatibility(self):
        """
        Validates compatibility between the convolutional and normalization layers.
        """
        if self.normalization_class not in CONVOLUTION_COMPATIBILITY_MAP.get(self.convolution_dim)["normalization"]:
            expected_norm_layers = CONVOLUTION_COMPATIBILITY_MAP.get(self.convolution_dim, {}).get("normalization", [])
            raise ValueError(
                f"Incompatible normalization layer for {self.normalization_class}. "
                f"Expected one of {expected_norm_layers} for convolution dimension {self.convolution_dim}."
            )

    def _validate_grouping_configuration(self):
        """
        Validates the grouping configuration for the convolutional layer.
        """
        if self.groups <= 0:
            raise ValueError('Groups must be a positive integer.')
        if self.input_dim % self.groups != 0:
            raise ValueError('Input dimension must be divisible by groups.')
        if self.output_dim % self.groups != 0:
            raise ValueError('Output dimension must be divisible by groups.')


    def _configure_dropout(self):
        """
        Configure the dropout layer based on the convolution class and dropout probability.

        Returns:
        --------
        nn.Module or None
            Dropout layer if dropout probability is greater than 0, otherwise None
        """
        if self.dropout_probability > 0:
            return CONVOLUTION_COMPATIBILITY_MAP[self.convolutional_dim]["dropout"](p=self.dropout_probability)
        return None


    def _initialize_weights(self):
        """
        Initialize the weights for the base and spline convolutional layers.
        """
        for conv_layer in self.base_convolutional_layers:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.spline_convolutional_layers:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')


    def _configure_convolutional_layer(self, input_dim, output_dim, groups, kernel_size, stride, padding, dilation):
        """
        Initialize convolutional layers for the base and spline transformations.
        :param input_dim:
        :param output_dim:
        :param groups:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :return:
        """
        convolutional_class = CONVOLUTION_COMPATIBILITY_MAP[self.convolution_dim]["convolution"]
        return nn.ModuleList([
            convolutional_class(
                    in_channels = input_dim,
                    out_channels = output_dim,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    dilation = dilation,
                    groups = 1,
                    bias = False)
            for _ in range(groups)
        ])


    def _configure_base_convolutional_layers(self):
        """
        Initialize the base convolutional layers for the KANConvNDLayer.
        """
        return self._configure_convolutional_layer(
            input_dim=self.input_dim // self.groups,
            output_dim=self.output_dim // self.groups,
            groups=self.groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation_rate
        )


    def _configure_spline_convolutional_layers(self):
        """
        Initialize the spline convolutional layers for the KANConvNDLayer.
        """
        spline_input_channels = (self.grid_size + self.spline_order) * self.input_dim // self.groups

        return self._configure_convolutional_layer(
            input_dim=spline_input_channels,
            output_dim=self.output_dim // self.groups,
            groups=self.groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation_rate
        )


    def _configure_normalization_layers(self):
        """
        Initialize normalization layers for the KANConvNDLayer.
        """
        return nn.ModuleList([
            self.normalization_class(
                self.output_dim // self.groups,
                **self.normalization_kwargs)
            for _ in range(self.groups)])


    def _configure_prelu_activation_functions(self):
        """
        Initialize PReLU activation functions for the KANConvNDLayer.
        """
        return nn.ModuleList([
            nn.PReLU()
            for _ in range(self.groups)])


    def _create_spline_grid(self, grid_size: int, spline_order: int) -> torch.Tensor:
        """
        Create a grid for spline transformation based on the specified range and spline order.

        Args:
            grid_size (int): The number of points in the grid.
            spline_order (int): The order of the spline.

        Returns:
            torch.Tensor: A tensor containing the spline grid.
        """
        grid_min = self.grid_range[0]
        grid_max = self.grid_range[1]
        grid_min_offset = spline_order
        grid_max_offset = spline_order
        total_points = grid_size + 2 * spline_order + 1  # Total number of points in the grid

        grid_spacing = (self.grid_range[1] - self.grid_range[0]) / grid_size

        spline_grid = torch.linspace(
            grid_min - grid_spacing * grid_min_offset,
            grid_max + grid_spacing * grid_max_offset,
            total_points,
            dtype=torch.float32
        )
        return spline_grid


    def _forward_spline(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        """
        Forward pass for a single group, applying base and spline transformations.

        Args:
        ----
        x : torch.Tensor
            Input tensor for the current group.
        group_index : int
            Index of the group to process.

        Returns:
        -------
        torch.Tensor
            Output tensor after processing the group with base and spline transformations.
        """
        # Base activation and convolution
        base_output = self.base_convolutional_layers[group_index](self.base_activation_function(x))

        # Spline transformation
        spline_output = self._apply_spline_transformation(x, group_index)

        # Combine, normalize, activate, and apply dropout if defined
        combined_output = base_output + spline_output
        normalized_output = self.normalization_layers[group_index](combined_output)
        activated_output = self.prelu_activations[group_index](normalized_output)

        if self.dropout_layer:
            activated_output = self.dropout_layer(activated_output)

        return activated_output


    def _apply_spline_transformation(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        """
        Applies spline interpolation transformation on the input.

        Args:
        ----
        x : torch.Tensor
            Input tensor.
        group_index : int
            Index of the group to process.

        Returns:
        -------
        torch.Tensor
            Output tensor after spline transformation.
        """
        x_expanded = x.unsqueeze(-1)
        expanded_grid = self._expand_grid(x)
        initial_bases = initialize_basis(x_expanded, expanded_grid)

        refined_bases = self._refine_basis(initial_bases, x_expanded, expanded_grid)

        # Flatten and apply convolution
        flattened_bases = refined_bases.moveaxis(-1, 2).flatten(1, 2)
        return self.spline_convolutional_layers[group_index](flattened_bases)

    def _expand_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expands the grid to match the input tensor dimensions.

        Args:
        ----
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Expanded grid tensor.
        """
        target_shape = x.shape[1:] + self.spline_grid.shape
        return self.spline_grid.view(*[1] * (self.convolution_dim + 1) + [-1]).expand(target_shape).to(x.device)

    def _refine_basis(self, bases: torch.Tensor, x_expanded: torch.Tensor, expanded_grid: torch.Tensor) -> torch.Tensor:
        """
        Refines basis functions based on the spline order.

        Args:
        ----
        bases : torch.Tensor
            Initial basis tensor.
        x_expanded : torch.Tensor
            Expanded input tensor.
        expanded_grid : torch.Tensor
            Expanded grid tensor.

        Returns:
        -------
        torch.Tensor
            Refined basis tensor.
        """
        for order in range(1, self.spline_order + 1):
            left_bounds, right_bounds = get_interval_bounds(expanded_grid, order)
            interval_delta = calculate_interval_delta(left_bounds, right_bounds)
            next_left, next_right = get_next_grid_slices(expanded_grid, order)

            bases = ((x_expanded - left_bounds) / interval_delta * bases[..., :-1]) + \
                    ((next_right - x_expanded) / (next_right - next_left) * bases[..., 1:])
        return bases


def initialize_basis(x_expanded: torch.Tensor, expanded_grid: torch.Tensor) -> torch.Tensor:
    """
    Initializes the basis functions for spline interpolation.

    Args:
    ----
    x_expanded : torch.Tensor
        Expanded input tensor.
    expanded_grid : torch.Tensor
        Expanded grid tensor.

    Returns:
    -------
    torch.Tensor
        Initial basis tensor.
    """
    left_grid_slice = expanded_grid[..., :-1]
    right_grid_slice = expanded_grid[..., 1:]
    return ((x_expanded >= left_grid_slice) & (x_expanded < right_grid_slice)).to(x_expanded.dtype)


def get_interval_bounds(expanded_grid: torch.Tensor, order: int) -> tuple:
    """Get left and right bounds for the current order interval."""
    return expanded_grid[..., :-(order + 1)], expanded_grid[..., order:-1]


def calculate_interval_delta(left_bounds: torch.Tensor, right_bounds: torch.Tensor) -> torch.Tensor:
    """Calculate the interval delta, ensuring non-zero values."""
    return torch.where(right_bounds == left_bounds, torch.ones_like(right_bounds), right_bounds - left_bounds)


def get_next_grid_slices(expanded_grid: torch.Tensor, order: int) -> tuple:
    """Get next left and right slices of the grid for the order refinement."""
    return expanded_grid[..., 1:-order], expanded_grid[..., order + 1:]


if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 32, 32)

    convolution_dim = 2
    normalization_class = nn.BatchNorm2d
    input_dim = 3
    output_dim = 64
    spline_order = 3
    kernel_size = 3
    groups = 1
    padding = 1
    stride = 1
    dilation_rate = 1
    grid_size = 5
    base_activation_function = nn.GELU
    grid_range = [-1, 1]
    dropout_probability = 0.0
    normalization_kwargs = {}

    kan_convolutional_layer = KANConvNDLayer(
        convolution_dim=convolution_dim,
        normalization_class=normalization_class,
        input_dim=input_dim,
        output_dim=output_dim,
        spline_order=spline_order,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=stride,
        dilation_rate=dilation_rate,
        grid_size=grid_size,
        base_activation_function=base_activation_function,
        grid_range=grid_range,
        dropout_probability=dropout_probability,
        **normalization_kwargs
    )

    output_tensor = kan_convolutional_layer(input_tensor)

    # Output tensor shape: torch.Size([1, 64, 32, 32])
    print(f"Output tensor shape: {output_tensor.shape}")
