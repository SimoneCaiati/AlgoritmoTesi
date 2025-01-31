import matplotlib.pyplot as plt 
from matplotlib import animation 
from colour import Color 

class Visualizer:
    
    def __init__(self, position, timestamp, media_dir):
        self.position = position
        self.timestamp = timestamp
        self.mediaDir = media_dir


    def plot_path(self, title="", n_sections=10, c1="green", c2="blue"):
            section_width = int(len(self.timestamp)/n_sections)
            colors = list(Color(c1).range_to(Color(c2), n_sections))
            plt.figure(figsize=(10, 10)) 
            ax = plt.axes(projection ='3d') 
            x = self.position[:,0]
            y = self.position[:,1]
            z = self.position[:,2]
            for i in range(n_sections):
                i_start = i*section_width
                i_end = i_start+section_width
                ax.plot3D(x[i_start:i_end], y[i_start:i_end], z[i_start:i_end], color=colors[i].hex)
            ax.scatter3D(x[0], y[0], z[0], s=50, color=c1, label="Start")
            ax.scatter3D(x[-1], y[-1], z[-1], s=50, color=c2, label="End")
            if title:
                ax.set_title(title)
            ax.set_xlabel("m")
            ax.set_ylabel("m")
            ax.set_zlabel("m")
            plt.legend()
            plt.savefig(self.mediaDir + "/" + "path.png")
            plt.show() 
            plt.close()
    
    def animate_path(self, length_sec=15, fps=4, c1="green", c2="blue"):
        samples_per_frame = int(len(self.timestamp)/length_sec/fps)
        n_frames = int(length_sec * fps)
        colors = list(Color(c1).range_to(Color(c2), n_frames))
        figure = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        x = self.position[:,0]
        y = self.position[:,1]
        z = self.position[:,2]
        ax.scatter3D(x[0], y[0], z[0], s=50, color=c1, label="Start")
        def update(frame):
            i_start = frame*samples_per_frame
            i_end = i_start+samples_per_frame
            ax.plot3D(x[i_start:i_end], y[i_start:i_end], z[i_start:i_end], color=colors[frame].hex)
            if i_end<6000:
                title=str(round(i_end/100,2))+" s"
            elif i_end<360000:
                title=str(round(i_end/6000,2))+" min"
            else: 
                title=str(round(i_end/360000,2))+" h"
            ax.set_title(title)
            if frame==n_frames-1:
                ax.scatter3D(x[-1], y[-1], z[-1], s=50, color=c2, label="End") 
        anim = animation.FuncAnimation(figure, update,
                                frames=n_frames,
                                interval=1000/fps,
                                repeat=False)
        ax.set_xlabel("m")
        ax.set_ylabel("m")
        ax.set_zlabel("m")
        plt.legend()
        anim.save(self.mediaDir + "/" +"animation.gif", writer=animation.PillowWriter(fps))
        plt.show()
        plt.close()
        
    def plot_acceleration_data(self, Acc, file_name):
        figure, ax = plt.subplots()
        ax.plot(self.timestamp, Acc[:, 0], "tab:red", label="X")
        ax.plot(self.timestamp, Acc[:, 1], "tab:green", label="Y")
        ax.plot(self.timestamp, Acc[:, 2], "tab:blue", label="Z")
        ax.set_title(file_name)
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.set_xlabel("Time (s)")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.mediaDir + "/" + file_name +".png")
        plt.show()
        plt.close()
    
    def plot_euler_angles(self, Orient):
        plt.plot(self.timestamp, Orient[:, 0], "tab:red", label="Pitch")
        plt.plot(self.timestamp, Orient[:, 1], "tab:green", label="Roll")
        plt.plot(self.timestamp, Orient[:, 2], "tab:blue", label="Yaw")
        plt.title("Euler angles")
        plt.xlabel("t [s]")
        plt.ylabel("rad/s")
        plt.grid()
        plt.legend()
        plt.savefig(self.mediaDir + "/" +"euler_angles.png")
        plt.show()
        plt.close()

