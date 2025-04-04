import random
import math
import torch
import torch.nn as nn


class Polygon(object):
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
            [n, x1, y1, x2, y2, ..., xn, yn, L1, L2, ..., Ln, A1, A2, ..., An].

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
        tokenised = [self.n]
        for x, y in self.vertices:
            tokenised.extend([round(x, 3), round(y, 3)])
        tokenised.extend([round(l, 3) for l in self.lengths])
        tokenised.extend([round(a, 2) for a in self.angles])
        return tokenised

    @classmethod
    def from_tokenised(cls, tokenised):
        """
        Creates a Polygon instance from a tokenised flat list.

        Parameters
        ----------
        tokenised : list

        Returns
        -------
        Polygon
            A new instance of Polygon constructed from the tokenised data.
        """
        n = int(tokenised[0])
        vertices_flat = tokenised[1 : 1 + 2 * n]
        vertices = []
        for i in range(0, len(vertices_flat), 2):
            vertices.append((vertices_flat[i], vertices_flat[i + 1]))
        lengths = tokenised[1 + 2 * n : 1 + 3 * n]
        angles = tokenised[1 + 3 * n :]
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
    computing their side lengths and interior angles, and tokenizing the result
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

    def _process_sample(self, tokens, mask):
        """
        Given a tokenized polygon and its binary mask, returns context set split.
        """
        x, y = [], []
        for token, m in zip(tokens, mask):
            if m == 1:
                x.append(token)
            else:
                y.append(token)
        return x, y

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
                tokens = tokens + [0] * pad_len
            else:
                tokens = tokens[:pad_length]
            padded.append(torch.tensor(tokens, dtype=torch.float))
        return torch.stack(padded)

    def generate_random_mask(self, token_length):
        # Force the first token to be 1.
        mask = [1]
        for _ in range(token_length - 1):
            mask.append(random.choice([0, 1]))
        # Ensure at least one token (besides the first) is masked.
        if all(m == 1 for m in mask[1:]):
            idx = random.randint(1, token_length - 1)
            mask[idx] = 0
        return mask

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

    def generate_masked_polygon_batch(self, num_context=None):
        """
        Generates a batch of polygons

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

        for _ in range(self.batch_size):

            tokens_list = []

            # Choose a fixed number of sides for this sample
            n = random.randint(self.min_num_sides, self.max_num_sides)

            # Generate the target polygon and its tokenised form.
            target_poly = self.generate_polygon(n)
            target_tokens = target_poly.to_tokenised()
            total_tokens = len(target_tokens)

            # For testing, use a deterministic mask (e.g., first half context, second half target)
            if self.testing:
                mask = [1] * (1 + 2 * n) + [0] * (total_tokens - (1 + 2 * n))
            else:
                mask = self.generate_random_mask(total_tokens)
            # Ensure first token is always context.
            mask[0] = 1

            context_x_list, context_y_list = [], []

            for _ in range(num_context):
                poly = self.generate_polygon(n)
                tokens = poly.to_tokenised()
                tokens_list.append(tokens)
                cx, cy = self._process_sample(tokens, mask)
                context_x_list.append(cx)
                context_y_list.append(cy)

            tx, ty = self._process_sample(target_tokens, mask)

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
        )
