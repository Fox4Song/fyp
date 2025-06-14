import random
import math
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.errors import GEOSException
import torch
import torch.nn as nn

from .tokens import SEP_VERTS, SEP_LENS, SEP_ANGS, EOS_TOKEN, MASK_TOKEN


class Polygon:
    """
    Represents a convex polygon with its vertices, side lengths, and interior angles.

    The polygon data includes:
      - n: the number of sides (vertices)
      - vertices: a list of (x, y) coordinates in counterclockwise order
      - lengths: a list of side lengths between consecutive vertices
      - angles: a list of interior angles (in degrees) at each vertex

    Methods
    -------
    to_tokenised():
        Converts the polygon data into a tokenised flat list in the format:
            [n, <VERTS>, x1, y1, x2, y2, ..., xn, yn, <LENS>, L1, L2, ..., Ln, <ANGS>, A1, A2, ..., An].

    from_tokenised(tokenised):
        Class method that creates a Polygon instance from a tokenised flat list.
    """

    def __init__(
        self, vertices, lengths, angles, center=(5, 5), radius=3, max_num_sides=12
    ):
        self._n = len(vertices)
        self._vertices = vertices
        self._lengths = lengths
        self._angles = angles
        self._center = center
        self._radius = radius
        self._max_num_sides = max_num_sides

    @property
    def n(self):
        return self._n

    @property
    def vertices(self):
        return self._vertices

    @property
    def lengths(self):
        return self._lengths

    @property
    def angles(self):
        return self._angles

    def _normalise_n(self):
        """
        Returns n / max_num_sides, so that n ∈ [1, max_num_sides] maps into [0,1].
        """
        return float(self.n) / float(self._max_num_sides)

    @staticmethod
    def _unnormalise_n(n, max_num_sides):
        """
        Converts a normalised value back to the original number of sides.
        """
        return int(n * max_num_sides)

    def _normalise_vertices(self):
        """
        For each (x,y) ∈ [cₓ - r, cₓ + r] x [cᵧ - r, cᵧ + r],
        map it into [0,1]x[0,1] by:
            x_norm = (x - (cₓ - r)) / (2r)  = (x - cₓ) / (2r) + 0.5
            y_norm = (y - (cᵧ - r)) / (2r)  = (y - cᵧ) / (2r) + 0.5
        Returns a flat list: [x1_norm, y1_norm, x2_norm, y2_norm, ...].
        """
        cx, cy = self._center
        r = self._radius

        normalised = []
        for x, y in self.vertices:
            x_norm = (x - cx) / (2.0 * r) + 0.5
            y_norm = (y - cy) / (2.0 * r) + 0.5
            normalised.extend((x_norm, y_norm))
        return normalised

    @staticmethod
    def _unnormalise_vertices(vertices_norm, center, radius):
        """
        Invert: x_norm = (x - cx) / (2r) + 0.5  →  x = (x_norm - 0.5) * 2r + cx,
               y_norm = (y - cy) / (2r) + 0.5  →  y = (y_norm - 0.5) * 2r + cy.

        vertices_norm is a flat list [x1_norm, y1_norm, x2_norm, y2_norm, ..., xn_norm, yn_norm].
        Returns a list of vertex tuples [(x1, y1), (x2, y2), ...].
        """
        cx, cy = center
        coords = []
        for i in range(0, len(vertices_norm), 2):
            x_norm = vertices_norm[i]
            y_norm = vertices_norm[i + 1]
            # invert the mapping
            x_raw = (x_norm - 0.5) * (2.0 * radius) + cx
            y_raw = (y_norm - 0.5) * (2.0 * radius) + cy
            coords.append((x_raw, y_raw))
        return coords

    def _normalise_lengths(self):
        """
        Each side-length L_i is at most 2⋅radius. We divide by (2⋅radius)
        so that all lengths lie in (0,1]. Returns a list [L1/(2r), L2/(2r), ...].
        """
        diameter = 2.0 * self._radius
        return [L / diameter for L in self.lengths]

    @staticmethod
    def _unnormalise_lengths(lengths_norm, radius):
        """
        Invert: L_norm = L_raw / (2r)  →  L_raw = L_norm * (2r).
        lengths_norm is a list [L1_norm, L2_norm, …, Ln_norm].
        Returns [L1_raw, L2_raw, ..., Ln_raw].
        """
        diameter = 2.0 * radius
        return [L_norm * diameter for L_norm in lengths_norm]

    def _normalise_angles(self):
        """
        Each interior angle A_i ∈ (0, 180). We divide by 180 to map it into (0,1].
        Returns a list [A1/180, A2/180, ...].
        """
        return [angle / 180.0 for angle in self.angles]

    @staticmethod
    def _unnormalise_angles(angles_norm):
        """
        Invert: A_norm = A_raw / 180  →  A_raw = A_norm * 180.
        angles_norm is a list [A1_norm, A2_norm, …, An_norm].
        Returns [A1_raw, A2_raw, …, An_raw].
        """
        return [A_norm * 180.0 for A_norm in angles_norm]

    def to_tokenised(self):
        """
        Converts the polygon into a tokenised flat list.

        Returns
        -------
        list
            The tokenised representation of the polygon.
        """
        tokenised = []
        tokenised.append(self._normalise_n())
        tokenised.append(SEP_VERTS)
        tokenised.extend(self._normalise_vertices())
        tokenised.append(SEP_LENS)
        tokenised.extend(self._normalise_lengths())
        tokenised.append(SEP_ANGS)
        tokenised.extend(self._normalise_angles())
        tokenised.append(EOS_TOKEN)
        return tokenised

    @classmethod
    def from_tokenised(cls, tokenised, n, center, radius, max_num_sides):
        """
        Creates a Polygon instance from a tokenised flat list.

        Parameters
        ----------
        tokenised : list
            The tokenised representation of the polygon.

        n : int
            The number of sides (vertices) of the polygon.

        Returns
        -------
        Polygon
            A new instance of Polygon constructed from the tokenised data.
        """
        vertices_flat = tokenised[2 : 2 + 2 * n]
        vertices = cls._unnormalise_vertices(vertices_flat, center, radius)
        lengths = tokenised[3 + 2 * n : 3 + 3 * n]
        lengths = cls._unnormalise_lengths(lengths, radius)
        angles = tokenised[4 + 3 * n : -1]
        angles = cls._unnormalise_angles(angles)
        return cls(vertices, lengths, angles, center, radius, max_num_sides)

    def __repr__(self):
        return "Polygon(n=%d, vertices=%s, lengths=%s, angles=%s)" % (
            self.n,
            self.vertices,
            self.lengths,
            self.angles,
        )


class PolygonSentenceReader(nn.Module):
    """
    Creates a dataset of polygon sentences by generating convex polygons,
    computing their side lengths and interior angles, and tokenising the result
    into a flat list.

    Each polygon sentence follows the grammar:
        [n, x1, y1, x2, y2, ..., xn, yn, L1, L2, ..., Ln, A1, A2, ..., An]
    where n is the number of vertices, (xi, yi) are the vertex coordinates (in counterclockwise order),
    Li are the side lengths, and Ai are the interior angles (in degrees).

    Parameters
    ----------
    batch_size : int
        Number of polygon sentences to sample.

    min_num_sides : int
        Minimum number of sides (vertices) for the polygon.

    max_num_sides : int
        Maximum number of sides (vertices) for the polygon.

    center : tuple (optional)
        Center of the circle used for generating polygon vertices (default is (5, 5)).

    radius : float (optional)
        Radius of the circle used for generating polygon vertices (default is 3).
    """

    def __init__(
        self,
        batch_size,
        max_num_context,
        max_seq_len,
        min_num_sides,
        max_num_sides,
        exclude_sides=[6, 7, 11, 12],
        center=(5, 5),
        radius=3,
        testing=False,
    ):
        super(PolygonSentenceReader, self).__init__()
        self.batch_size = batch_size
        self.max_num_context = max_num_context
        self.max_seq_len = max_seq_len
        self.min_num_sides = min_num_sides
        self.max_num_sides = max_num_sides
        self.exclude_sides = exclude_sides
        self.center = center
        self.radius = radius
        self.testing = testing

    def _generate_random_convex_polygon(self, n):
        """
        Generates a convex polygon by sampling n points on a circle centered at the specified center.

        The function samples n random angles uniformly between 0 and 2π, sorts them to ensure
        the vertices are in counterclockwise order, and then computes the corresponding (x, y)
        coordinates on a circle of the given radius.

        Parameters
        ----------
        n : int
            The number of vertices of the polygon.

        Returns
        -------
        list
            A list of vertices (x, y) representing the convex polygon in counterclockwise order.
        """
        center = self.center
        radius = self.radius

        # Generate n random angles between 0 and 2π, then sort them for counterclockwise order.
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(n)])

        # Compute (x, y) coordinates on the circle for each angle.
        vertices = [
            (
                round(center[0] + radius * math.cos(angle), 2),
                round(center[1] + radius * math.sin(angle), 2),
            )
            for angle in angles
        ]
        return vertices

    def _compute_side_lengths(self, vertices):
        """
        Computes the side lengths of a polygon given its vertices.

        Parameters
        ----------
        vertices : list
            A list of vertices (x, y) of the polygon in order.

        Returns
        -------
        list
            A list of side lengths for each edge of the polygon.
        """
        n = len(vertices)
        lengths = []
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            length = math.hypot(x2 - x1, y2 - y1)
            lengths.append(length)
        return lengths

    def _compute_interior_angles(self, vertices):
        """
        Computes the interior angles of a polygon from its vertices in degrees.

        Parameters
        ----------
        vertices : list
            A list of vertices (x, y) of the polygon in counterclockwise order.

        Returns
        -------
        list
            A list of interior angles (in degrees) for each vertex of the polygon.
        """
        n = len(vertices)
        angles = []
        for i in range(n):
            # Get previous, current, and next vertices (with wrap-around).
            prev = vertices[i - 1]
            curr = vertices[i]
            next = vertices[(i + 1) % n]

            # Compute vectors from the current vertex to the previous and next vertices.
            v1 = (prev[0] - curr[0], prev[1] - curr[1])
            v2 = (next[0] - curr[0], next[1] - curr[1])

            # Compute the angle using dot product.
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.hypot(*v1)
            mag2 = math.hypot(*v2)

            denom = mag1 * mag2

            # If the denominator is near zero, set the angle to 0 (or handle appropriately)
            if denom < 1e-6:
                angle_rad = 0.0
            else:
                # Clamp the cosine value to the range [-1, 1] to avoid math domain errors.
                cosine = max(min(dot / denom, 1.0), -1.0)
                angle_rad = math.acos(cosine)
            angles.append(math.degrees(angle_rad))
        return angles

    def _pad_batch(self, list_of_lists, pad_length):
        """
        Pads a list of token lists to a fixed length with zeros.

        Parameters
        ----------
        list_of_lists : list of lists
            Each inner list is a sequence of tokens (numbers).

        pad_length : int
            The desired fixed length for each sequence.

        Returns
        -------
        torch.Tensor
            A tensor of shape [B, pad_length] with each sequence padded with 0s.
        """
        padded = []
        for tokens in list_of_lists:
            tokens = list(tokens)
            pad_len = pad_length - len(tokens)
            if pad_len > 0:
                tokens = tokens + [0.0] * pad_len
            else:
                tokens = tokens[:pad_length]
            padded.append(torch.tensor(tokens, dtype=torch.float))
        return torch.stack(padded)

    def _generate_random_mask(self, token_length, probability):
        mask = torch.rand(token_length) < probability
        mask = mask.to(torch.int)
        return mask

    def _sample_random_transformation(self, t_type=None, eval=False):
        """
        Samples a random transformation type and its parameters.

        Returns
        -------
        tuple
            A tuple containing the transformation type and its parameters.
            The transformation type is one of "rotation", "scaling", or "translation".
            The parameters are a dictionary with the necessary values for the transformation.
        """
        if not t_type:
            t_type = random.choice(["rotation", "scaling", "translation"])

        if t_type == "rotation":
            params = {
                "angle": random.uniform(180, 270) if eval else random.uniform(15, 180)
            }
        elif t_type == "scaling":
            params = {
                "scale_x": (
                    random.uniform(1.5, 2.0) if eval else random.uniform(0.5, 1.5)
                ),
                "scale_y": (
                    random.uniform(1.5, 2.0) if eval else random.uniform(0.5, 1.5)
                ),
            }
        elif t_type == "translation":
            params = {
                "dx": random.uniform(2, 4) if eval else random.uniform(-2, 2),
                "dy": random.uniform(2, 4) if eval else random.uniform(-2, 2),
            }
        else:
            raise ValueError("Unknown Transformation Type")

        return t_type, params

    def _transform_polygon(self, polygon, t_type=None, parameters=None, eval=False):
        """
        Applies a transformation to the given polygon.

        Parameters
        ----------
        polygon : Polygon
            The original polygon to transform.

        t_type : str (optional)
            The type of transformation ("rotation", "scaling", or "translation").
            If None, a transformation is chosen at random.

        parameters : dict (optional)
            Parameters for the transformation. The expected keys depend on the transformation type:
            - "rotation": {"angle": float} (degrees)
            - "scaling": {"scale_x": float, "scale_y": float}
            - "translation": {"dx": float, "dy": float}

        Returns
        -------
        transformed_polygon : Polygon
            The polygon after applying the transformation.
        """
        if t_type is None:
            t_type = random.choice(["rotation", "scaling", "translation"])

        if t_type == "rotation":
            # Rotate around the polygon's centroid.
            if parameters is None:
                if eval:
                    angle = random.uniform(180.0, 270.0)
                else:
                    angle = random.uniform(15, 180)  # degrees
            else:
                angle = parameters["angle"]
            angle_rad = math.radians(angle)
            cx = sum(v[0] for v in polygon.vertices) / polygon.n
            cy = sum(v[1] for v in polygon.vertices) / polygon.n
            new_vertices = []
            for x, y in polygon.vertices:
                new_x = (
                    cx + math.cos(angle_rad) * (x - cx) - math.sin(angle_rad) * (y - cy)
                )
                new_y = (
                    cy + math.sin(angle_rad) * (x - cx) + math.cos(angle_rad) * (y - cy)
                )
                new_vertices.append((new_x, new_y))
        elif t_type == "scaling":
            # Scale around the polygon's centroid.
            if parameters is None:
                if eval:
                    scalex = random.uniform(1.5, 2.0)
                    scaley = random.uniform(1.5, 2.0)
                else:
                    scalex = random.uniform(0.5, 1.5)
                    scaley = random.uniform(0.5, 1.5)
            else:
                scalex = parameters["scale_x"]
                scaley = parameters["scale_y"]
            cx = sum(v[0] for v in polygon.vertices) / polygon.n
            cy = sum(v[1] for v in polygon.vertices) / polygon.n
            new_vertices = []
            for x, y in polygon.vertices:
                new_x = cx + scalex * (x - cx)
                new_y = cy + scaley * (y - cy)
                new_vertices.append((new_x, new_y))
        elif t_type == "translation":
            # Translate by a random vector.
            if parameters is None:
                if eval:
                    dx = random.uniform(-2, 2)
                    dy = random.uniform(-2, 2)
                else:
                    dx = random.uniform(-4, 4)
                    dy = random.uniform(-4, 4)
            else:
                dx = parameters["dx"]
                dy = parameters["dy"]
            new_vertices = [(x + dx, y + dy) for (x, y) in polygon.vertices]

        new_lengths = self._compute_side_lengths(new_vertices)
        new_angles = self._compute_interior_angles(new_vertices)
        transformed_polygon = Polygon(new_vertices, new_lengths, new_angles)
        return transformed_polygon

    def generate_polygon(self, n=None):
        """
        Generates a polygon by constructing a convex polygon, computing its side lengths and interior angles.

        Returns
        -------
        Polygon
            A Polygon object representing the generated convex polygon.
        """
        # Randomly choose the number of sides within the specified range.
        if n is None:
            allowed = [
                i
                for i in range(self.min_num_sides, self.max_num_sides + 1)
                if i not in self.exclude_sides
            ]
            n = random.choice(allowed)
        vertices = self._generate_random_convex_polygon(n)
        lengths = self._compute_side_lengths(vertices)
        angles = self._compute_interior_angles(vertices)

        return Polygon(vertices, lengths, angles, max_num_sides=self.max_num_sides)

    def generate_polygon_batch_few_shot_masked_completion_task(
        self, num_context=None, n=None, mask_cfg=None
    ):
        """
        Gnerates a batch of Polygons for Few-Shot Masked Completion Tasks

        Given a partial derivation (e.g., masked SIDE, LENGTH, ANGLE, etc.),
        predict missing components such as LENGTHS and/or ANGLES.

        Returns
        -------
        context_x : torch.Tensor [B, num_context, x_size]
            The x values of the context points

        context_y : torch.Tensor [B, num_context, y_size]
            The y values of the context points

        target_x : torch.Tensor [B, num_target, x_size]
            The x values of the target points

        target_y : torch.Tensor [B, num_target, y_size]
            The y values of the target points

        true_target_polygons : list
            List of true target polygons.

        max_total_tokens : int
            Maximum number of tokens (after padding) in the batch.

        num_context : int
            Number of context points in the batch.

        context_masks : torch.Tensor [B, max_seq_len]
            The mask for the context points, where 1 indicates a masked point and 0 indicates an unmasked point.
        """

        if num_context is None:
            num_context = torch.randint(low=3, high=self.max_num_context + 1, size=(1,))

        num_target = torch.randint(low=2, high=self.max_num_context + 1, size=(1,))

        context_x, context_y = [], []
        context_masks = []
        target_x, target_y = [], []
        total_tokens_list = []
        true_target_polygons = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            if n is None:
                allowed = [
                    i
                    for i in range(self.min_num_sides, self.max_num_sides + 1)
                    if i not in self.exclude_sides
                ]
                n = random.choice(allowed)

            # Generate the target polygon and its tokenised form.
            target_poly = self.generate_polygon(n)
            target_tokens = target_poly.to_tokenised()
            total_tokens = len(target_tokens)

            # For testing, use a deterministic mask (e.g., mask only angles)
            if self.testing and mask_cfg is not None:
                num_target = 1
                if mask_cfg["type"] == "angle":
                    mask = (
                        [0] * (4 + 3 * n) + [1] * (total_tokens - (4 + 3 * n + 1)) + [0]
                    )
                elif mask_cfg["type"] == "length":
                    mask = (
                        [0] * (3 + 2 * n) + [1] * n + [0] * (total_tokens - (3 + 3 * n))
                    )
                elif mask_cfg["type"] == "vertex":
                    mask = [0] * 2 + [1] * (2 * n) + [0] * (total_tokens - (2 + 2 * n))
                else:
                    mask = [1] * total_tokens
                if "p" in mask_cfg:
                    p = mask_cfg["p"]
                    one_positions = [i for i, v in enumerate(mask) if v == 1]
                    num_to_keep = int(len(one_positions) * p)
                    keep_positions = set(random.sample(one_positions, num_to_keep))
                    mask = [
                        1 if i in keep_positions else 0 for i in range(total_tokens)
                    ]
            else:
                # Mask 15%
                mask = self._generate_random_mask(total_tokens, 0.15)
                mask = mask.tolist()

            context_x_list, context_y_list = [], []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)
                cx = [MASK_TOKEN if m == 1 else t for t, m in zip(tokens, mask)]
                context_x_list.append(cx)
                context_y_list.append(tokens)

            tx_list, ty_list = context_x_list.copy(), context_y_list.copy()
            target_poly_list = []

            for _ in range(num_target):
                target_poly = self.generate_polygon(n)
                target_poly_list.append(target_poly)
                target_tokens = target_poly.to_tokenised()
                tx = [MASK_TOKEN if m == 1 else t for t, m in zip(target_tokens, mask)]
                tx_list.append(tx)
                ty_list.append(target_tokens)

            # Pad each list into a tensor.
            context_x_pad = self._pad_batch(context_x_list, self.max_seq_len)
            context_y_pad = self._pad_batch(context_y_list, self.max_seq_len)
            target_x_pad = self._pad_batch(tx_list, self.max_seq_len)
            target_y_pad = self._pad_batch(ty_list, self.max_seq_len)
            context_mask = self._pad_batch([mask], self.max_seq_len)

            context_x.append(context_x_pad)
            context_y.append(context_y_pad)
            target_x.append(target_x_pad)
            target_y.append(target_y_pad)
            context_masks.append(context_mask)
            total_tokens_list.append(total_tokens)
            true_target_polygons.append(target_poly_list)

        # Stack individual samples to create batch tensors.
        context_x = torch.stack(context_x)  # [B, num_context, max_seq_len]
        context_y = torch.stack(context_y)
        target_x = torch.stack(target_x)  # [B, num_target, max_seq_len]
        target_y = torch.stack(target_y)
        context_masks = torch.stack(context_masks)  # [B, 1, max_seq_len]

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            total_tokens_list,
            true_target_polygons,
            self.max_seq_len,
            num_context,
            num_target,
            context_masks,
        )

    def generate_polygon_batch_few_shot_completion_task(
        self, num_context=None, n=None, num_pred_angles=None
    ):
        """
        Gnerates a batch of Polygons for Few-Shot Completion Tasks

        Given a partial derivation (without ONE component e.g the SIDES),
        predict missing components such as LENGTHS and/or ANGLES.

        Returns
        -------
        context_x : torch.Tensor [B, num_context, x_size]
            The x values of the context points

        context_y : torch.Tensor [B, num_context, y_size]
            The y values of the context points

        target_x : torch.Tensor [B, num_target, x_size]
            The x values of the target points

        target_y : torch.Tensor [B, num_target, y_size]
            The y values of the target points

        true_target_polygons : list
            List of true target polygons.

        max_total_tokens : int
            Maximum number of tokens (after padding) in the batch.

        num_context : int
            Number of context points in the batch.
        """

        if num_context is None:
            num_context = torch.randint(low=3, high=self.max_num_context + 1, size=(1,))

        if self.testing:
            num_target = 1
        else:
            num_target = torch.randint(low=2, high=self.max_num_context + 1, size=(1,))

        context_x, context_y = [], []
        context_masks = []
        target_x, target_y = [], []
        total_tokens_list = []
        true_target_polygons = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            if n is None:
                allowed = [
                    i
                    for i in range(self.min_num_sides, self.max_num_sides + 1)
                    if i not in self.exclude_sides
                ]
                n = random.choice(allowed)

            if num_pred_angles is None:
                num_query_angles = n - random.randint(1, n)
            else:
                num_query_angles = n - num_pred_angles

            context_x_list, context_y_list = [], []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)

                # Split tokens into context_x and context_y
                # - context_x contains the polygon sequence up to and including <SEP_ANG> and (len_angles - num_pred_angles) angles
                # - context_y contains the polygon sequence representing num_pred_angles angles
                cx = tokens[: 4 + 3 * n + num_query_angles]
                cy = tokens[4 + 3 * n + num_query_angles : -1]

                context_x_list.append(cx)
                context_y_list.append(cy)

            tx_list, ty_list = context_x_list.copy(), context_y_list.copy()
            target_poly_list = []

            for _ in range(num_target):
                # Generate the target polygon and its tokenised form.
                target_poly = self.generate_polygon(n)
                target_poly_list.append(target_poly)
                target_tokens = target_poly.to_tokenised()
                total_tokens = len(target_tokens)
                tx = target_tokens[: 4 + 3 * n + num_query_angles]
                ty = target_tokens[4 + 3 * n + num_query_angles : -1]
                tx_list.append(tx)
                ty_list.append(ty)

            mask = [1] * len(ty)

            # Pad each list into a tensor.
            context_x_pad = self._pad_batch(context_x_list, self.max_seq_len)
            # x_size = 3 + 4 * max_sides
            # y_size = max_sides
            y_size = (self.max_seq_len - 3) // 4
            context_y_pad = self._pad_batch(context_y_list, y_size)
            target_x_pad = self._pad_batch(tx_list, self.max_seq_len)
            target_y_pad = self._pad_batch(ty_list, y_size)
            context_mask = self._pad_batch([mask], y_size)

            context_x.append(context_x_pad)
            context_y.append(context_y_pad)
            target_x.append(target_x_pad)
            target_y.append(target_y_pad)
            context_masks.append(context_mask)
            total_tokens_list.append(total_tokens)
            true_target_polygons.append(target_poly_list)

        # Stack individual samples to create batch tensors.
        context_x = torch.stack(context_x)  # [B, num_context, max_seq_len]
        context_y = torch.stack(context_y)
        target_x = torch.stack(target_x)  # [B, num_target, max_seq_len]
        target_y = torch.stack(target_y)
        context_masks = torch.stack(context_masks)  # [B, 1, y_size]

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            total_tokens_list,
            true_target_polygons,
            self.max_seq_len,
            num_context,
            num_target,
            context_masks,
        )

    def generate_polygon_batch_few_shot_transformation_task(
        self,
        num_context=None,
        n=None,
        transformation_type=None,
        next_transformation_type=None,
        eval=False,
    ):
        """
        Gnerates a batch of Polygons for Few-Shot Transformation Tasks

        Rotate, scale, or translate a polygon and predict
        the new properties, with optional OOD parameter sampling.

        Parameters
        ----------
        transformation_type : str
            The type of transformation to apply to the polygon.
            Can be one of 'rotation', 'translation', or 'scaling'.

        next_transformation_type : str
            The successive type of transformation to apply to the polygon.
            Can be one of 'rotation', 'translation', or 'scaling'.

        eval: bool
            If True, the function will generate a batch of polygons
            with the same transformation type for evaluation.

        Returns
        -------
        context_x : torch.Tensor [B, num_context, x_size]
            The x values of the context points

        context_y : torch.Tensor [B, num_context, y_size]
            The y values of the context points

        target_x : torch.Tensor [B, num_target, x_size]
            The x values of the target points

        target_y : torch.Tensor [B, num_target, y_size]
            The y values of the target points

        true_target_polygons : list
            List of true target polygons.

        true_transformed_polygons : list
            List of true transformed polygons.

        max_total_tokens : int
            Maximum number of tokens (after padding) in the batch.

        num_context : int
            Number of context points in the batch.
        """

        if num_context is None:
            num_context = torch.randint(low=3, high=self.max_num_context + 1, size=(1,))

        if self.testing:
            num_target = 1
        else:
            num_target = torch.randint(low=2, high=self.max_num_context + 1, size=(1,))

        context_x, context_y = [], []
        target_x, target_y = [], []
        total_tokens_list = []
        true_target_polygons = []
        true_transformed_polygons = []
        context_masks = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            if n is None:
                allowed = [
                    i
                    for i in range(self.min_num_sides, self.max_num_sides + 1)
                    if i not in self.exclude_sides
                ]
                n = random.choice(allowed)

            # Sample random transformation
            transformation_type, params = self._sample_random_transformation(
                transformation_type, eval=eval
            )
            if next_transformation_type is not None:
                if next_transformation_type == "random":
                    next_transformation_type, next_transformation_params = (
                        self._sample_random_transformation(None, eval=eval)
                    )
                else:
                    next_transformation_type, next_transformation_params = (
                        self._sample_random_transformation(
                            next_transformation_type, eval=eval
                        )
                    )

            context_x_list, context_y_list = [], []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)
                transformed_context_poly = self._transform_polygon(
                    poly, transformation_type, params
                )
                if next_transformation_type is not None:
                    transformed_context_poly = self._transform_polygon(
                        transformed_context_poly,
                        next_transformation_type,
                        next_transformation_params,
                    )
                transformed_tokens_context = transformed_context_poly.to_tokenised()
                context_x_list.append(tokens)
                context_y_list.append(transformed_tokens_context)

            tx_list, ty_list = context_x_list.copy(), context_y_list.copy()
            target_poly_list, transformed_target_poly_list = [], []

            for _ in range(num_target):
                # Generate the target polygon and its tokenised form.
                target_poly = self.generate_polygon(n)
                target_poly_list.append(target_poly)
                transformed_poly = self._transform_polygon(
                    target_poly, transformation_type, params
                )
                if next_transformation_type is not None:
                    transformed_poly = self._transform_polygon(
                        transformed_poly,
                        next_transformation_type,
                        next_transformation_params,
                    )
                transformed_target_poly_list.append(transformed_poly)
                target_tokens = target_poly.to_tokenised()
                target_trans_tokens = transformed_poly.to_tokenised()
                total_tokens = len(target_trans_tokens)
                tx = target_tokens
                ty = target_trans_tokens
                tx_list.append(tx)
                ty_list.append(ty)

            mask = [1] * len(ty)

            # Pad each list into a tensor.
            context_x_pad = self._pad_batch(context_x_list, self.max_seq_len)
            context_y_pad = self._pad_batch(context_y_list, self.max_seq_len)
            target_x_pad = self._pad_batch(tx_list, self.max_seq_len)
            target_y_pad = self._pad_batch(ty_list, self.max_seq_len)
            mask_pad = self._pad_batch([mask], self.max_seq_len)

            context_x.append(context_x_pad)
            context_y.append(context_y_pad)
            target_x.append(target_x_pad)
            target_y.append(target_y_pad)
            context_masks.append(mask_pad)
            total_tokens_list.append(total_tokens)
            true_target_polygons.append(target_poly_list)
            true_transformed_polygons.append(transformed_target_poly_list)

        # Stack individual samples to create batch tensors.
        context_x = torch.stack(context_x)  # [B, num_context, max_seq_len]
        context_y = torch.stack(context_y)
        target_x = torch.stack(target_x)  # [B, num_target, max_seq_len]
        target_y = torch.stack(target_y)
        context_masks = torch.stack(context_masks)  # [B, 1, max_seq_len]

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            context_masks,
            total_tokens_list,
            true_target_polygons,
            true_transformed_polygons,
            self.max_seq_len,
            num_context,
            num_target,
        )

    def generate_polygon_batch_few_shot_composition_task(
        self, num_context=None, n=None, operation_type=None
    ):
        """
        Few-shot composition tasks with resampling on invalid compositions:
        - Ensures composed shape is a single, valid, non-empty Polygon.

        Parameters
        ----------
        operation_type : str
            The type of operation to apply to the polygons.
            Can be one of 'union', 'intersection'.

        Returns
        -------
        context_x : torch.Tensor [B, num_context, x_size]
            The x values of the context points

        context_y : torch.Tensor [B, num_context, y_size]
            The y values of the context points

        target_x : torch.Tensor [B, num_target, x_size]
            The x values of the target points

        target_y : torch.Tensor [B, num_target, y_size]
            The y values of the target points

        true_target_polygons : list
            List of true target polygons.

        max_total_tokens : int
            Maximum number of tokens (after padding) in the batch.

        num_context : int
            Number of context points in the batch.
        """

        if num_context is None:
            num_context = torch.randint(low=3, high=self.max_num_context + 1, size=(1,))

        if self.testing:
            num_target = 1
        else:
            num_target = torch.randint(low=2, high=self.max_num_context + 1, size=(1,))

        all_ctx_x, all_ctx_y = [], []
        all_qx, all_qy = [], []
        total_tokens_list = []
        true_target_polygons = []
        true_query_pairs = []  # # list of (Polygon1, Polygon2) for each query
        context_masks = []

        for _ in range(self.batch_size):

            # Context set
            ctx_inputs, ctx_targets = [], []
            for _ in range(num_context):
                # resample until valid composition
                while True:
                    p1 = self.generate_polygon(n)
                    p2 = self.generate_polygon(n)
                    sp1 = ShapelyPolygon(p1.vertices)
                    sp2 = ShapelyPolygon(p2.vertices)
                    if not sp1.is_valid or not sp2.is_valid:
                        continue
                    op = operation_type or random.choice(["union", "intersection"])
                    # check validity
                    try:
                        if op == "union":
                            raw = sp1.union(sp2)
                            sc_shape = raw.convex_hull
                        else:
                            sc_shape = sp1.intersection(sp2)
                    except GEOSException:
                        continue
                    if (
                        not sc_shape.is_empty
                        and sc_shape.is_valid
                        and sc_shape.geom_type == "Polygon"
                    ):
                        break
                coords = list(sc_shape.exterior.coords)[:-1]
                comp_poly = Polygon(
                    [(x, y) for x, y in coords],
                    self._compute_side_lengths(coords),
                    self._compute_interior_angles(coords),
                    max_num_sides=self.max_num_sides,
                )
                inp_tokens = p1.to_tokenised() + p2.to_tokenised()
                tgt_tokens = comp_poly.to_tokenised()
                ctx_inputs.append(inp_tokens)
                ctx_targets.append(tgt_tokens)

            tx_list, ty_list = ctx_inputs.copy(), ctx_targets.copy()
            target_polygons_list, query_poly_list = [], []

            for _ in range(num_target):

                # Target set
                while True:
                    q1 = self.generate_polygon(n)
                    q2 = self.generate_polygon(n)
                    sp1 = ShapelyPolygon(q1.vertices)
                    sp2 = ShapelyPolygon(q2.vertices)
                    if not sp1.is_valid or not sp2.is_valid:
                        continue
                    op = operation_type or random.choice(["union", "intersection"])
                    try:
                        if op == "union":
                            raw = sp1.union(sp2)
                            sc_shape = raw.convex_hull
                        else:
                            sc_shape = sp1.intersection(sp2)
                    except GEOSException:
                        continue
                    if (
                        not sc_shape.is_empty
                        and sc_shape.is_valid
                        and sc_shape.geom_type == "Polygon"
                    ):
                        break
                query_poly_list.append((q1, q2))
                coords = list(sc_shape.exterior.coords)[:-1]
                query_poly = Polygon(
                    [(x, y) for x, y in coords],
                    self._compute_side_lengths(coords),
                    self._compute_interior_angles(coords),
                    max_num_sides=self.max_num_sides,
                )
                q_inp = q1.to_tokenised() + q2.to_tokenised()
                q_tgt = query_poly.to_tokenised()
                tx_list.append(q_inp)
                ty_list.append(q_tgt)

                target_polygons_list.append(query_poly)

            total_tokens_list.append(len(q_tgt))

            # compute context mask
            mask = [1] * len(q_tgt)

            # pad and collect
            ctx_x_pad = self._pad_batch(ctx_inputs, self.max_seq_len)
            ctx_y_pad = self._pad_batch(ctx_targets, self.max_seq_len)
            qx_pad = self._pad_batch(tx_list, self.max_seq_len)
            qy_pad = self._pad_batch(ty_list, self.max_seq_len)
            mask_pad = self._pad_batch([mask], self.max_seq_len)

            all_ctx_x.append(ctx_x_pad)
            all_ctx_y.append(ctx_y_pad)
            all_qx.append(qx_pad)
            all_qy.append(qy_pad)
            true_target_polygons.append(target_polygons_list)
            true_query_pairs.append(query_poly_list)
            context_masks.append(mask_pad)

        # stack
        context_x = torch.stack(all_ctx_x)
        context_y = torch.stack(all_ctx_y)
        target_x = torch.stack(all_qx)
        target_y = torch.stack(all_qy)
        context_masks = torch.stack(context_masks)

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            context_masks,
            total_tokens_list,
            true_target_polygons,
            true_query_pairs,
            self.max_seq_len,
            num_context,
            num_target,
        )

    def generate_masked_polygon_batch(self):
        """
        Generates a batch of masked polygon token sequences for masked language modelling
        pretraining.

        For each polygon:
        - A tokenised sequence is generated.
        - A random binary mask is produced (1: token kept, 0: token masked). The first token is always kept.
        - The input sequence is created by replacing masked tokens with a special MASK token (here -1.0).
        - The label sequence contains the original token for masked positions and 0.0 for unmasked positions.
        - An additional mlm_mask tensor is produced indicating positions on which the loss should be computed (1 for masked positions, 0 otherwise).

        Returns
        -------
        input_batch : torch.Tensor [B, max_seq_len]
            Batch of masked input token sequences.

        label_batch : torch.Tensor [B, max_seq_len]
            Batch of label sequences containing original tokens for masked positions.

        mlm_mask_batch : torch.Tensor [B, max_seq_len]
            Batch of binary masks indicating masked token positions.
        """

        input_sequences = []
        label_sequences = []
        mlm_masks = []
        attention_masks = []

        for _ in range(self.batch_size):
            paragraph_tokens = []
            # Generate several polygon sentences and separate them by EOS_TOKEN.
            num_sentences = random.randint(3, self.max_num_context)
            for _ in range(num_sentences):
                poly = self.generate_polygon()
                tokens = poly.to_tokenised()
                paragraph_tokens.extend(tokens)

            # Mask 15%
            mask = self._generate_random_mask(len(tokens), 0.15)

            input_seq = []
            label_seq = []
            mlm_mask = []  # 1 if token is masked (loss is computed), else 0.

            input_seq = [MASK_TOKEN if m else token for token, m in zip(tokens, mask)]
            label_seq = [token if m else 0.0 for token, m in zip(tokens, mask)]
            mlm_mask = [1 if m else 0 for m in mask]

            # Pad sequences to fixed length self.max_seq_len.
            if len(input_seq) < self.max_seq_len:
                pad_length = self.max_seq_len - len(input_seq)
                input_seq = input_seq + [0.0] * pad_length
                label_seq = label_seq + [0.0] * pad_length
                mlm_mask = mlm_mask + [0] * pad_length
            else:
                input_seq = input_seq[: self.max_seq_len]
                label_seq = label_seq[: self.max_seq_len]
                mlm_mask = mlm_mask[: self.max_seq_len]

            attention_mask = [1 if token != 0.0 else 0 for token in input_seq]

            input_sequences.append(torch.tensor(input_seq, dtype=torch.float))
            label_sequences.append(torch.tensor(label_seq, dtype=torch.float))
            mlm_masks.append(torch.tensor(mlm_mask, dtype=torch.float))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.float))

        input_batch = torch.stack(input_sequences)  # [B, max_seq_len]
        label_batch = torch.stack(label_sequences)  # [B, max_seq_len]
        mlm_mask_batch = torch.stack(mlm_masks)  # [B, max_seq_len]
        attention_mask_batch = torch.stack(attention_masks)  # [B, max_seq_len]

        return input_batch, label_batch, mlm_mask_batch, attention_mask_batch

    def generate_causal_polygon_batch(self):
        """
        Generates a batch of token sequences for next-token (causal) language modeling.

        Returns
        -------
        input_batch : torch.Tensor [B, max_seq_len]
            Batch of input token sequences (each shifted right by one from label_seq).

        label_batch : torch.Tensor [B, max_seq_len]
            Batch of target token sequences (the “next” token at each position).

        attention_mask_batch : torch.Tensor [B, max_seq_len]
            Binary mask (1 for real tokens in input_seq, 0 for padding).
        """

        input_sequences = []
        label_sequences = []
        attention_masks = []

        for _ in range(self.batch_size):
            # Build a long token list by concatenating several polygons + EOS
            target_len = random.randint(256, self.max_seq_len + 1)
            paragraph_tokens = []
            while len(paragraph_tokens) < target_len:
                poly = self.generate_polygon()
                tokens = poly.to_tokenised()
                paragraph_tokens.extend(tokens)

            # Enforce length = max_seq_len + 1 via truncation or padding
            total_len = self.max_seq_len + 1
            if len(paragraph_tokens) >= total_len:
                seq = paragraph_tokens[:total_len]
            else:
                pad_len = total_len - len(paragraph_tokens)
                seq = paragraph_tokens + [0.0] * pad_len  # assume 0.0 is pad token

            # Split into input / label
            input_seq = seq[:-1]  # length = max_seq_len
            label_seq = seq[1:]  # length = max_seq_len

            # Attention mask for padding in input_seq
            attention_mask = [1 if tok != 0.0 else 0 for tok in input_seq]

            # Collect as tensors
            input_sequences.append(torch.tensor(input_seq, dtype=torch.float))
            label_sequences.append(torch.tensor(label_seq, dtype=torch.float))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.float))

        # Stack into [B, max_seq_len]
        input_batch = torch.stack(input_sequences)
        label_batch = torch.stack(label_sequences)
        attention_mask_batch = torch.stack(attention_masks)

        return input_batch, label_batch, attention_mask_batch

    def generate_causal_transformation_polygon_batch(self, mixing_ratio=0.05):
        """
        Generates a batch of token sequences for next-token (causal) language modeling,
        injecting transformation sentence (original→transformed) randomly interleaved within
        each paragraph with the given mixing ratio.

        Returns
        -------
        input_batch : torch.Tensor [B, max_seq_len]
            Batch of input token sequences (each shifted right by one from label_seq).

        label_batch : torch.Tensor [B, max_seq_len]
            Batch of target token sequences (the “next” token at each position).

        attention_mask_batch : torch.Tensor [B, max_seq_len]
            Binary mask (1 for real tokens in input_seq, 0 for padding).
        """

        input_sequences = []
        label_sequences = []
        attention_masks = []

        for _ in range(self.batch_size):
            # Determine if inject transformation sentence
            inject = random.random() < mixing_ratio
            # Build a long token list by concatenating several polygons + EOS
            target_len = random.randint(256, self.max_seq_len + 1)
            sentences = []
            sentence_len = 0
            while sentence_len < target_len:
                poly = self.generate_polygon()
                tokens = poly.to_tokenised()
                sentences.append(tokens)
                sentence_len += len(tokens)

            if inject:
                orig = self.generate_polygon()
                trans = self._transform_polygon(orig)
                orig = orig.to_tokenised()
                trans = trans.to_tokenised()
                pos = random.randint(0, len(sentences))
                sentences.insert(pos, orig)
                sentences.insert(pos + 1, trans)

            paragraph_tokens = []
            for s in sentences:
                paragraph_tokens.extend(s)

            # Enforce length = max_seq_len + 1 via truncation or padding
            total_len = self.max_seq_len + 1
            if len(paragraph_tokens) >= total_len:
                seq = paragraph_tokens[:total_len]
            else:
                pad_len = total_len - len(paragraph_tokens)
                seq = paragraph_tokens + [0.0] * pad_len  # assume 0.0 is pad token

            # Split into input / label
            input_seq = seq[:-1]  # length = max_seq_len
            label_seq = seq[1:]  # length = max_seq_len

            # Attention mask for padding in input_seq
            attention_mask = [1 if tok != 0.0 else 0 for tok in input_seq]

            # Collect as tensors
            input_sequences.append(torch.tensor(input_seq, dtype=torch.float))
            label_sequences.append(torch.tensor(label_seq, dtype=torch.float))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.float))

        # Stack into [B, max_seq_len]
        input_batch = torch.stack(input_sequences)
        label_batch = torch.stack(label_sequences)
        attention_mask_batch = torch.stack(attention_masks)

        return input_batch, label_batch, attention_mask_batch

    def generate_causal_polygon_batch_few_shot_masked_completion_task(
        self, num_context=None, mask_cfg=None
    ):
        """
        Generates a batch of Polygons for explicit Few-Shot Masked Completion Tasks
        in a next-token (causal) language modeling way.

        Given a number of context example [x1,y1,x2,y2,...,xn,yn] and a target,
        predict each next token in the sequence.

        Returns
        -------
        input_batch : torch.Tensor [B, max_seq_len]
            Batch of input token sequences (each shifted right by one from label_seq).

        label_batch : torch.Tensor [B, max_seq_len]
            Batch of target token sequences (the “next” token at each position).

        attention_mask_batch : torch.Tensor [B, max_seq_len]
            Binary mask (1 for real tokens in input_seq, 0 for padding).

        target_mask_batch : torch.Tensor [B, max_seq_len]
            Binary mask (1 for target tokens, 0 for non-target tokens).
        """

        if num_context is None:
            num_context = torch.randint(low=1, high=self.max_num_context + 1, size=(1,))

        input_sequences = []
        label_sequences = []
        attention_masks = []
        target_masks = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            allowed = [
                i
                for i in range(self.min_num_sides, self.max_num_sides + 1)
                if i not in self.exclude_sides
            ]
            n = random.choice(allowed)

            # Generate the target polygon and its tokenised form.
            target_poly = self.generate_polygon(n)
            target_tokens = target_poly.to_tokenised()
            total_tokens = len(target_tokens)

            # For testing, use a deterministic mask (e.g., mask only angles)
            if self.testing and mask_cfg is not None:
                num_target = 1
                if mask_cfg["type"] == "angle":
                    mask = (
                        [0] * (4 + 3 * n) + [1] * (total_tokens - (4 + 3 * n - 1)) + [0]
                    )
                elif mask_cfg["type"] == "length":
                    mask = (
                        [0] * (3 + 2 * n) + [1] * n + [0] * (total_tokens - (3 + 3 * n))
                    )
                elif mask_cfg["type"] == "vertex":
                    mask = [0] * 2 + [1] * (2 * n) + [0] * (total_tokens - (2 + 2 * n))
                else:
                    mask = [1] * total_tokens
                if "p" in mask_cfg:
                    p = mask_cfg["p"]
                    one_positions = [i for i, v in enumerate(mask) if v == 1]
                    num_to_keep = int(len(one_positions) * p)
                    keep_positions = set(random.sample(one_positions, num_to_keep))
                    mask = [
                        1 if i in keep_positions else 0 for i in range(total_tokens)
                    ]
            else:
                # Mask 15%
                mask = self._generate_random_mask(total_tokens, 0.15)
                mask = mask.tolist()

            paragraph_tokens = []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)
                cx = [MASK_TOKEN if m == 1 else t for t, m in zip(tokens, mask)]
                paragraph_tokens.extend(cx)
                cy = [t for t, m in zip(tokens, mask) if m == 1]
                paragraph_tokens.extend(cy)
                paragraph_tokens.append(EOS_TOKEN)

            tx = [MASK_TOKEN if m == 1 else t for t, m in zip(target_tokens, mask)]
            paragraph_tokens.extend(tx)
            context_query_len = len(paragraph_tokens)
            ty = [t for t, m in zip(target_tokens, mask) if m == 1]
            paragraph_tokens.extend(ty)
            paragraph_tokens.append(EOS_TOKEN)
            target_mask = [0] * context_query_len + [1] * len(ty) + [0]
            assert len(paragraph_tokens) == len(target_mask)

            # Enforce length = max_seq_len + 1 via truncation or padding
            total_len = self.max_seq_len + 1
            if len(paragraph_tokens) >= total_len:
                seq = paragraph_tokens[:total_len]
                target_mask = target_mask[:total_len]
            else:
                pad_len = total_len - len(paragraph_tokens)
                seq = paragraph_tokens + [0.0] * pad_len  # assume 0.0 is pad token
                target_mask = target_mask + [0] * pad_len

            # Split into input / label
            input_seq = seq[:-1]  # length = max_seq_len
            label_seq = seq[1:]  # length = max_seq_len

            target_mask = target_mask[1:]  # length = max_seq_len

            # Attention mask for padding in input_seq
            attention_mask = [1 if tok != 0.0 else 0 for tok in input_seq]

            # Collect as tensors
            input_sequences.append(torch.tensor(input_seq, dtype=torch.float))
            label_sequences.append(torch.tensor(label_seq, dtype=torch.float))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.float))
            target_masks.append(torch.tensor(target_mask, dtype=torch.float))

        # Stack into [B, max_seq_len]
        input_batch = torch.stack(input_sequences)
        label_batch = torch.stack(label_sequences)
        attention_mask_batch = torch.stack(attention_masks)
        target_mask_batch = torch.stack(target_masks)

        return input_batch, label_batch, attention_mask_batch, target_mask_batch

    def causal_polygon_batch_few_shot_completion_task(
        self, num_context=None, num_pred_angles=None
    ):
        """
        Generates a batch of Polygons for explicit Few-Shot Angle Completion Tasks
        in a next-token (causal) language modeling way.

        Given a number of context example [x1,y1,x2,y2,...,xn,yn] and a target,
        predict each next token in the sequence.

        Returns
        -------
        input_batch : torch.Tensor [B, max_seq_len]
            Batch of input token sequences (each shifted right by one from label_seq).

        label_batch : torch.Tensor [B, max_seq_len]
            Batch of target token sequences (the “next” token at each position).

        attention_mask_batch : torch.Tensor [B, max_seq_len]
            Binary mask (1 for real tokens in input_seq, 0 for padding).

        target_mask_batch : torch.Tensor [B, max_seq_len]
            Binary mask (1 for target tokens, 0 for non-target tokens).
        """

        if num_context is None:
            num_context = torch.randint(low=1, high=self.max_num_context + 1, size=(1,))

        input_sequences = []
        label_sequences = []
        attention_masks = []
        target_masks = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            allowed = [
                i
                for i in range(self.min_num_sides, self.max_num_sides + 1)
                if i not in self.exclude_sides
            ]
            n = random.choice(allowed)

            if num_pred_angles is None:
                num_query_angles = n - random.randint(1, n)
            else:
                num_query_angles = n - num_pred_angles

            paragraph_tokens = []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)

                # Split tokens into context_x and context_y
                # - context_x contains the polygon sequence up to and including <SEP_ANG> and (len_angles - num_pred_angles) angles
                # - context_y contains the polygon sequence representing num_pred_angles angles
                cx = tokens[: 4 + 3 * n + num_query_angles]
                cy = tokens[4 + 3 * n + num_query_angles : -1]

                paragraph_tokens.extend(cx)
                paragraph_tokens.extend(cy)

            # Generate the target polygon and its tokenised form.
            target_poly = self.generate_polygon(n)
            target_tokens = target_poly.to_tokenised()
            tx = target_tokens[: 4 + 3 * n + num_query_angles]
            ty = target_tokens[4 + 3 * n + num_query_angles : -1]
            paragraph_tokens.extend(tx)
            context_query_len = len(paragraph_tokens)
            paragraph_tokens.extend(ty)
            target_mask = [0] * context_query_len + [1] * len(ty)
            assert len(paragraph_tokens) == len(target_mask)

            # Enforce length = max_seq_len + 1 via truncation or padding
            total_len = self.max_seq_len + 1
            if len(paragraph_tokens) >= total_len:
                seq = paragraph_tokens[:total_len]
                target_mask = target_mask[:total_len]
            else:
                pad_len = total_len - len(paragraph_tokens)
                seq = paragraph_tokens + [0.0] * pad_len  # assume 0.0 is pad token
                target_mask = target_mask + [0] * pad_len

            # Split into input / label
            input_seq = seq[:-1]  # length = max_seq_len
            label_seq = seq[1:]  # length = max_seq_len

            target_mask = target_mask[1:]  # length = max_seq_len

            # Attention mask for padding in input_seq
            attention_mask = [1 if tok != 0.0 else 0 for tok in input_seq]

            # Collect as tensors
            input_sequences.append(torch.tensor(input_seq, dtype=torch.float))
            label_sequences.append(torch.tensor(label_seq, dtype=torch.float))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.float))
            target_masks.append(torch.tensor(target_mask, dtype=torch.float))

        # Stack into [B, max_seq_len]
        input_batch = torch.stack(input_sequences)
        label_batch = torch.stack(label_sequences)
        attention_mask_batch = torch.stack(attention_masks)
        target_mask_batch = torch.stack(target_masks)

        return input_batch, label_batch, attention_mask_batch, target_mask_batch
