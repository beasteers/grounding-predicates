import typing
import pathlib

# import sys
# sys.path.append("/scratch/data/repos/grounding-predicates")

import numpy as np
import symbolic

from apps import hand_detector
from env import twentybn
from gpred import video_utils, dnf_utils
import config


# import sys
# sys.path.append("..")

def display_video_grid(
    labels: twentybn.dataset.Labels,
    action_instances: typing.List[typing.List[int]],
    path: pathlib.Path,
    num_rows: int = 5
):
    """Displays 3 x N grid of videos.
    
    Args:
        labels: 20BN labels.
        action_instances: List of video ids per action.
        path: Path of videos.
        num_rows: Number of rows to display per batch.
    """
    from IPython.display import clear_output
    import ipywidgets as widgets
    
    next_button = widgets.Button(description="Next")
    
    def assign_button_handler(id_action: int):
        """Assigns click handler to 'Next' button."""
        
        SIZE_BATCH = 3 * num_rows
        num_examples = len(action_instances[id_action])
        idx_example_start = 0
        
        def show_next_video_callback(b: widgets.Button):
            """Called on button click to display next video grid."""
            nonlocal idx_example_start
            with output:
                clear_output()

                print(f"\n{labels.actions[id_action].template}")
                print(f"Examples {idx_example_start}..{idx_example_start + SIZE_BATCH - 1} out of {num_examples}\n")

                idx_examples = range(idx_example_start, idx_example_start + SIZE_BATCH)
                id_videos = [action_instances[id_action][idx_example] for idx_example in idx_examples]

                video_utils.display_video_grid(id_videos, path, labels=[labels.videos[id_video].action_name for id_video in id_videos])

                idx_example_start += SIZE_BATCH
        
        next_button._click_handlers.callbacks = []
        next_button.on_click(show_next_video_callback)

    input_action = widgets.BoundedIntText(value=0, min=0, max=len(labels.actions), description="Action index:")
    output = widgets.interactive_output(assign_button_handler, {"id_action": input_action})

    return widgets.VBox([widgets.HBox([input_action, next_button]), output])

