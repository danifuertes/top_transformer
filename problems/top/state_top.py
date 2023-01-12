import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F


class StateTOP(NamedTuple):

    # FIXED INPUTS
    # Depots + loc
    coords: torch.Tensor

    # Prize of each node
    prize: torch.Tensor

    # Max length a tour should have when arriving at each node: max_length = input['max_length'] - d(end_depot, node)
    max_length: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and prizes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # STATE INFO
    # Previous visited node
    prev_a: list

    # Mask of visited nodes
    visited_: torch.Tensor

    # Length of the tours
    lengths: list

    # Current coordinates
    cur_coord: list

    # Current collected prize
    cur_total_prize: torch.Tensor

    # Keeps track of step
    i: torch.Tensor

    # Index of end depot
    end_ids: torch.Tensor

    # Number of agents for the TOP
    num_agents: int

    @property
    def visited(self):
        """Returns a mask that informs about the nodes that have already been visited."""
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self):
        """Distance matrix."""
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            cur_total_prize=self.cur_total_prize[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8, num_agents=1):
        """Initialize TOP state."""

        # Inputs
        depot = input['depot']
        depot2 = input['depot2'] if 'depot2' in input else input['depot']
        loc = input['loc']
        prize = input['prize']
        max_length = input['max_length']

        # Dimensions
        batch_size, n_loc, _ = loc.size()
        graph_size = n_loc + 2

        # Coordinates = loc + depots
        coords = torch.cat((depot[:, None, :], loc, depot2[:, None, :]), -2)

        # Add prize of initial and end depot, which are 0
        prize = F.pad(prize, (1, 0), mode='constant', value=0)
        prize = torch.cat((prize, torch.zeros((batch_size, 1), device=loc.device)), dim=1)

        # Max length already considers the distance to return to the depot. Subtract epsilon for numeric stability
        max_length = max_length[:, None] - (depot2[:, None, :] - coords).norm(p=2, dim=-1) - 1e-6

        # Length of the tours
        lengths = [torch.zeros(batch_size, 1, device=loc.device) for _ in range(num_agents)]

        # Batch indexes
        ids = torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None]

        # Index of previous nodes
        prev_a = [torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device) for _ in range(num_agents)]

        # Mask of visited nodes
        visited_ = (
            torch.zeros(
                batch_size, 1, graph_size,
                dtype=torch.uint8, device=loc.device
            )
            if visited_dtype == torch.uint8
            else torch.zeros(batch_size, 1, (graph_size + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
        )

        # Current coordinates (start in initial depot)
        cur_coord = [depot[:, None, :] for _ in range(num_agents)]

        # Current prize (starts being 0)
        cur_total_prize = torch.zeros(batch_size, 1, device=loc.device)

        # Count the number of steps/iterations
        i = torch.zeros(1, dtype=torch.int64, device=loc.device)

        # Index of end depot
        end_ids = graph_size - 1

        # Create initial state
        return StateTOP(coords=coords,
                        prize=prize,
                        max_length=max_length,
                        lengths=lengths,
                        ids=ids,
                        prev_a=prev_a,
                        visited_=visited_,
                        cur_coord=cur_coord,
                        cur_total_prize=cur_total_prize,
                        i=i,
                        end_ids=end_ids,
                        num_agents=num_agents)

    def get_remaining_length(self, agent_id):
        """Get the length that an agent can still use to travel."""
        # max_length[:, end_ids] is max length arriving at end depot
        return self.max_length[self.ids, self.end_ids] - self.lengths[agent_id]

    def get_final_cost(self):
        """Cost is the negative of the collected prize since we want to maximize collected prize."""
        assert self.all_finished()
        return -self.cur_total_prize

    def update(self, selected):
        """Update state with new nodes visited."""
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state using each agent info
        prev_a, cur_coord, lengths, cur_total_prize, visited_ = [], [], [], self.cur_total_prize, self.visited
        for k in range(self.num_agents):

            # Current node index to prev_a
            current = selected[:, None, k]  # Add dimension for step
            prev_a.append(current)

            # Add the length
            cur_coord.append(self.coords[self.ids, current])
            lengths.append(self.lengths[k] + (cur_coord[k] - self.cur_coord[k]).norm(p=2, dim=-1))

            # Add the collected prize
            cur_total_prize += self.prize[self.ids, current]

            # Update mask of visited nodes
            if self.visited_.dtype == torch.uint8:
                # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
                # Add one dimension since we write a single value
                visited_ = visited_ | visited_.scatter(-1, current[:, :, None], 1)
            else:
                # This works, by check_unset=False it is allowed to set the depot visited a second a time
                visited_ = visited_ | mask_long_scatter(visited_, current, check_unset=False)

        # Update state
        return self._replace(prev_a=prev_a,
                             visited_=visited_,
                             lengths=lengths,
                             cur_coord=cur_coord,
                             cur_total_prize=cur_total_prize,
                             i=self.i + 1)

    def update_gamma(self, selected, agent_id):
        """Update state with new nodes visited."""
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state using agent info
        prev_a = self.prev_a.copy()
        cur_coord = self.cur_coord.copy()
        lengths = self.lengths
        cur_total_prize = self.cur_total_prize
        visited_ = self.visited

        # Current node index to prev_a
        current = selected[:, None]  # Add dimension for step
        prev_a[agent_id] = current

        # Add the length
        cur_coord[agent_id] = self.coords[self.ids, current]
        prev_coord = self.coords[self.ids, self.prev_a[agent_id]]
        lengths[agent_id] = self.lengths[agent_id] + (cur_coord[agent_id] - prev_coord).norm(p=2, dim=-1)

        # Add the collected prize
        cur_total_prize += self.prize[self.ids, current]

        # Update mask of visited nodes
        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = visited_ | visited_.scatter(-1, current[:, :, None], 1)
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = visited_ | mask_long_scatter(visited_, current, check_unset=False)

        # Update state
        return self._replace(prev_a=prev_a,
                             visited_=visited_,
                             lengths=lengths,
                             cur_coord=cur_coord,
                             cur_total_prize=cur_total_prize,
                             i=self.i + 1)

    def all_finished(self):
        """All must be returned to depot (and at least 1 step since at start also prev_a == 0). This is more efficient
        than checking the mask."""
        finished = True
        for k in range(self.num_agents):
            finished = finished and (self.prev_a[k] == self.end_ids).all()
        return self.i.item() > 0 and finished

    def get_current_node(self, agent_id):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: list of (batch_size, num_steps) tensors with current nodes. len(prev_a) == num_agents.
        """
        return self.prev_a[agent_id]

    def get_mask(self, agent_id):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        # Check which nodes can be visited without exceeding the max_length constraint
        exceeds_length = (
            self.lengths[agent_id][:, :, None] +
            (self.coords[self.ids, :, :] - self.cur_coord[agent_id][:, :, None, :]).norm(p=2, dim=-1) >
            self.max_length[self.ids, :]
        )

        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # Cannot visit if already visited or if length that would be upon arrival is too large to return to depot
        # If the depot has already been visited then we cannot visit anymore
        visited_ = self.visited.to(exceeds_length.dtype)
        in_end_depot = (self.get_current_node(agent_id) == self.end_ids)[:, :, None].to(exceeds_length.dtype)
        mask = visited_ | in_end_depot | exceeds_length

        # Block initial depot
        mask[:, :, 0] = 1

        # End depot can always be visited
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        mask[:, :, self.end_ids] = 0
        # mask[self.ids.squeeze(1), :, self.get_current_node(agent_id).squeeze(1)]
        return mask

    def construct_solutions(self, actions):
        return actions
