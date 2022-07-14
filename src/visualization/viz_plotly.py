# Import data
import time
import numpy as np

from skimage import io
import plotly.graph_objects as go



# vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
# volume = vol.T
# r, c = volume[0].shape

# # Define frames
# import plotly.graph_objects as go
# nb_frames = 68

def plot_volume_mri(volume):
    nb_frames, r, c = volume.shape

    z_num = nb_frames - 1
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(z_num - 0.1 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames - 1 - k]),
        cmin=0, cmax=200
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(z_num-0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[z_num-1]),
        colorscale='Gray',
        cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
        ))


    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=600,
            height=600,
            scene=dict(
                        zaxis=dict(range=[-0.1, z_num], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )

    fig.show()

def plot_volume_regular(volume): 
    nb_frames, r, c = volume.shape

    x1 = np.linspace(0, c-1, c)  
    y1 = np.linspace(-5, r-1, r)  
    z1 = np.linspace(0, nb_frames-1, nb_frames)  

    X, Y, Z = np.meshgrid(x1, y1, z1) 

    values = volume 

    fig = go.Figure(data=go.Volume( 
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(), 
        value=values.flatten(), 
        opacity=0.1, 
        )) 

    fig.show()