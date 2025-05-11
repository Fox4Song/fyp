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

    def __init__(self, vertices, lengths, angles):
        self._n = len(vertices)
        self._vertices = vertices
        self._lengths = lengths
        self._angles = angles

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

    def to_tokenised(self):
        """
        Converts the polygon into a tokenised flat list.

        Returns
        -------
        list
            The tokenised representation of the polygon.
        """
        tokenised = [self.n, SEP_VERTS]
        for x, y in self.vertices:
            tokenised.extend([x, y])
        tokenised.append(SEP_LENS)
        tokenised.extend(self.lengths)
        tokenised.append(SEP_ANGS)
        tokenised.extend(self.angles)
        tokenised.append(EOS_TOKEN)
        return tokenised

    @classmethod
    def from_tokenised(cls, tokenised, n):
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
        vertices = [
            (vertices_flat[i], vertices_flat[i + 1])
            for i in range(0, len(vertices_flat), 2)
        ]
        lengths = tokenised[3 + 2 * n : 3 + 3 * n]
        angles = tokenised[4 + 3 * n : -1]
        return cls(vertices, lengths, angles)

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

    def _sample_random_transformation(self, t_type=None):
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
            params = {"angle": random.uniform(15, 180)}
        elif t_type == "scaling":
            params = {
                "scale_x": random.uniform(0.5, 1.5),
                "scale_y": random.uniform(0.5, 1.5),
            }
        elif t_type == "translation":
            params = {"dx": random.uniform(-2, 2), "dy": random.uniform(-2, 2)}
        else:
            raise ValueError("Unknown Transformation Type")

        return t_type, params

    def _transform_polygon(self, polygon, t_type=None, parameters=None):
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
                dx = random.uniform(-2, 2)
                dy = random.uniform(-2, 2)
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
            n = random.randint(self.min_num_sides, self.max_num_sides)
        vertices = self._generate_random_convex_polygon(n)
        lengths = self._compute_side_lengths(vertices)
        angles = self._compute_interior_angles(vertices)

        return Polygon(vertices, lengths, angles)

    def generate_polygon_batch_few_shot_masked_completion_task(self, num_context=None):
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

        context_x, context_y = [], []
        context_masks = []
        target_x, target_y = [], []
        total_tokens_list = []
        true_target_polygons = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            n = random.randint(self.min_num_sides, self.max_num_sides)

            # Generate the target polygon and its tokenised form.
            target_poly = self.generate_polygon(n)
            target_tokens = target_poly.to_tokenised()
            total_tokens = len(target_tokens)

            # For testing, use a deterministic mask (e.g., mask only angles)
            if self.testing:
                mask = [0] * (4 + 3 * n) + [1] * (total_tokens - (4 + 3 * n - 1)) + [0]
                # Hide vertices for visualisation
                # mask = [0]
                # for vert_idx in range(n):
                #     if vert_idx % 2 == 0:
                #         # hide both x and y of this vertex
                #         mask.extend([1, 1])
                #     else:
                #         # keep both x and y
                #         mask.extend([0, 0])
                # rest = total_tokens - len(mask)
                # mask.extend([0] * rest)
            else:
                # Mask 15%
                mask = self._generate_random_mask(total_tokens, 0.15)
                mask = mask.tolist()
            # TODO: REMOVE
            mask = [0] * (4 + 3 * n) + [1] * (total_tokens - (4 + 3 * n - 1)) + [0]

            context_x_list, context_y_list = [], []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)
                cx = [MASK_TOKEN if m == 1 else t for t, m in zip(tokens, mask)]
                context_x_list.append(cx)
                context_y_list.append(tokens)

            tx = [MASK_TOKEN if m == 1 else t for t, m in zip(target_tokens, mask)]
            ty = target_tokens

            # Pad each list into a tensor.
            context_x_pad = self._pad_batch(context_x_list, self.max_seq_len)
            context_y_pad = self._pad_batch(context_y_list, self.max_seq_len)
            target_x_pad = self._pad_batch([tx], self.max_seq_len)
            target_y_pad = self._pad_batch([ty], self.max_seq_len)
            context_mask = self._pad_batch([mask], self.max_seq_len)

            context_x.append(context_x_pad)
            context_y.append(context_y_pad)
            target_x.append(target_x_pad)
            target_y.append(target_y_pad)
            context_masks.append(context_mask)
            total_tokens_list.append(total_tokens)
            true_target_polygons.append(target_poly)

        # Stack individual samples to create batch tensors.
        context_x = torch.stack(context_x)  # [B, num_context, max_seq_len]
        context_y = torch.stack(context_y)
        target_x = torch.stack(target_x)  # [B, 1, max_seq_len]
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
            context_masks,
        )

    def generate_polygon_batch_few_shot_completion_task(self, num_context=None):
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

        context_x, context_y = [], []
        context_masks = []
        target_x, target_y = [], []
        total_tokens_list = []
        true_target_polygons = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            n = random.randint(self.min_num_sides, self.max_num_sides)

            # Generate the target polygon and its tokenised form.
            target_poly = self.generate_polygon(n)
            target_tokens = target_poly.to_tokenised()
            total_tokens = len(target_tokens)

            context_x_list, context_y_list = [], []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)

                # Split tokens into context_x and context_y
                # - context_x contains the polygon sequence up to and including <SEP_ANG>
                # - context_y contains the polygon sequence representing the angles
                cx = tokens[: 4 + 3 * n]
                cy = tokens[4 + 3 * n : -1]

                context_x_list.append(cx)
                context_y_list.append(cy)

            tx = target_tokens[: 4 + 3 * n]
            ty = target_tokens[4 + 3 * n : -1]

            mask = [1] * len(ty)

            # Pad each list into a tensor.
            context_x_pad = self._pad_batch(context_x_list, self.max_seq_len)
            y_size = (self.max_seq_len - 4) // 3
            context_y_pad = self._pad_batch(context_y_list, y_size)
            target_x_pad = self._pad_batch([tx], self.max_seq_len)
            target_y_pad = self._pad_batch([ty], y_size)
            context_mask = self._pad_batch([mask], y_size)

            context_x.append(context_x_pad)
            context_y.append(context_y_pad)
            target_x.append(target_x_pad)
            target_y.append(target_y_pad)
            context_masks.append(context_mask)
            total_tokens_list.append(total_tokens)
            true_target_polygons.append(target_poly)

        # Stack individual samples to create batch tensors.
        context_x = torch.stack(context_x)  # [B, num_context, max_seq_len]
        context_y = torch.stack(context_y)
        target_x = torch.stack(target_x)  # [B, 1, max_seq_len]
        target_y = torch.stack(target_y)

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            total_tokens_list,
            true_target_polygons,
            self.max_seq_len,
            num_context,
            context_masks,
        )

    def generate_polygon_batch_few_shot_transformation_task(
        self, num_context=None, transformation_type=None
    ):
        """
        Gnerates a batch of Polygons for Few-Shot Transformation Tasks

        Rotate, scale, or translate a polygon and predict
        the new properties

        Parameters
        ----------
        transformation_type : str
            The type of transformation to apply to the polygon.
            Can be one of 'rotation', 'translation', or 'scaling'.

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

        context_x, context_y = [], []
        target_x, target_y = [], []
        total_tokens_list = []
        true_target_polygons = []
        true_transformed_polygons = []

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            n = random.randint(self.min_num_sides, self.max_num_sides)

            # Sample random transformation
            transformation_type, params = self._sample_random_transformation(
                transformation_type
            )

            # Generate the target polygon and its tokenised form.
            target_poly = self.generate_polygon(n)
            transformed_poly = self._transform_polygon(
                target_poly, transformation_type, params
            )
            target_tokens = target_poly.to_tokenised()
            target_trans_tokens = transformed_poly.to_tokenised()
            total_tokens = len(target_trans_tokens)

            context_x_list, context_y_list = [], []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)
                transformed_context_poly = self._transform_polygon(
                    poly, transformation_type, params
                )
                transformed_tokens_context = transformed_context_poly.to_tokenised()
                context_x_list.append(tokens)
                context_y_list.append(transformed_tokens_context)

            tx = target_tokens
            ty = target_trans_tokens

            # Pad each list into a tensor.
            context_x_pad = self._pad_batch(context_x_list, self.max_seq_len)
            context_y_pad = self._pad_batch(context_y_list, self.max_seq_len)
            target_x_pad = self._pad_batch([tx], self.max_seq_len)
            target_y_pad = self._pad_batch([ty], self.max_seq_len)

            context_x.append(context_x_pad)
            context_y.append(context_y_pad)
            target_x.append(target_x_pad)
            target_y.append(target_y_pad)
            total_tokens_list.append(total_tokens)
            true_target_polygons.append(target_poly)
            true_transformed_polygons.append(transformed_poly)

        # Stack individual samples to create batch tensors.
        context_x = torch.stack(context_x)  # [B, num_context, max_seq_len]
        context_y = torch.stack(context_y)
        target_x = torch.stack(target_x)  # [B, 1, max_seq_len]
        target_y = torch.stack(target_y)

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            total_tokens_list,
            true_target_polygons,
            true_transformed_polygons,
            self.max_seq_len,
            num_context,
        )

    def generate_polygon_batch_few_shot_composition_task(
        self, num_context=None, operation_type=None
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
            num_context = torch.randint(3, self.max_num_context + 1, (1,)).item()

        all_ctx_x, all_ctx_y = [], []
        all_qx, all_qy = [], []
        total_tokens_list = []
        true_target_polygons = []
        true_query_pairs = []  # # list of (Polygon1, Polygon2) for each query

        for _ in range(self.batch_size):
            # Context set
            ctx_inputs, ctx_targets = [], []
            for _ in range(num_context):
                # resample until valid composition
                while True:
                    p1 = self.generate_polygon()
                    p2 = self.generate_polygon()
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
                )
                inp_tokens = p1.to_tokenised() + p2.to_tokenised()
                tgt_tokens = comp_poly.to_tokenised()
                ctx_inputs.append(inp_tokens)
                ctx_targets.append(tgt_tokens)

            # Target set
            while True:
                q1 = self.generate_polygon()
                q2 = self.generate_polygon()
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
            true_query_pairs.append((q1, q2))
            coords = list(sc_shape.exterior.coords)[:-1]
            query_poly = Polygon(
                [(x, y) for x, y in coords],
                self._compute_side_lengths(coords),
                self._compute_interior_angles(coords),
            )
            q_inp = q1.to_tokenised() + q2.to_tokenised()
            q_tgt = query_poly.to_tokenised()

            true_target_polygons.append(query_poly)
            total_tokens_list.append(len(q_tgt))

            # pad and collect
            ctx_x_pad = self._pad_batch(ctx_inputs, self.max_seq_len)
            ctx_y_pad = self._pad_batch(ctx_targets, self.max_seq_len)
            qx_pad = self._pad_batch([q_inp], self.max_seq_len)
            qy_pad = self._pad_batch([q_tgt], self.max_seq_len)

            all_ctx_x.append(ctx_x_pad)
            all_ctx_y.append(ctx_y_pad)
            all_qx.append(qx_pad)
            all_qy.append(qy_pad)

        # stack
        context_x = torch.stack(all_ctx_x)
        context_y = torch.stack(all_ctx_y)
        target_x = torch.stack(all_qx)
        target_y = torch.stack(all_qy)

        return (
            context_x,
            context_y,
            target_x,
            target_y,
            total_tokens_list,
            true_target_polygons,
            true_query_pairs,
            self.max_seq_len,
            num_context,
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
