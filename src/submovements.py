import numpy as np
import pyaldata
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def plot_hold_move_speed(trial,fig=None,max_speed=None,hold_thresh=None,hold_slice_fun=None,move_slice_fun=None,events_to_plot=None):
    if hold_slice_fun is None:
        hold_slice_fun = pyaldata.generate_epoch_fun(
                start_point_name='idx_goCueTime',
                rel_start = -0.3/trial['bin_size'],
                rel_end = 0,
                )
    if move_slice_fun is None:
        move_slice_fun = pyaldata.generate_epoch_fun(
                start_point_name='idx_goCueTime',
                end_point_name='idx_endTime',
                )

    if fig is None:
        fig = plt.figure()
    else:
        fig.clear()

    gs = mpl.gridspec.GridSpec(2,3,width_ratios=(4,1,1))
    pos_ax = fig.add_subplot(gs[0,0])
    sm_ax = fig.add_subplot(gs[0,1:])
    speed_ax = fig.add_subplot(gs[1,0],sharex=pos_ax)
    hold_hist_ax = fig.add_subplot(gs[1,1],sharey=speed_ax)
    move_hist_ax = fig.add_subplot(gs[1,2],sharey=speed_ax)
    # axs = fig.subplots(1,3,sharey=True,gridspec_kw={'width_ratios':(4,1,1)})
    axs = [speed_ax,hold_hist_ax,move_hist_ax]


    if 'hand_speed' not in trial:
        trial['hand_speed'] = np.linalg.norm(trial['hand_vel'],axis=1)

    hold_sig = trial['hand_speed'][hold_slice_fun(trial),...]
    move_sig = trial['hand_speed'][move_slice_fun(trial),...]
    bin_size = trial['bin_size']

    if max_speed is None:
        max_speed = np.percentile(abs(move_sig),99,axis=0)

    if hold_thresh is None:
        hold_thresh = np.percentile(abs(hold_sig),99,axis=0)

    move_time = bin_size*hold_sig.shape[0]
    end_time = move_time+bin_size*move_sig.shape[0]
    axs[0].plot(
        np.arange(0,move_time,bin_size)-move_time,
        hold_sig,
        c='k',
    )
    axs[0].plot(
        bin_size*np.arange(move_sig.shape[0]),
        move_sig,
        c='r',
    )
    axs[0].plot(
            np.zeros((2,)),
            [0,max_speed],
            '--g'
    )
    axs[0].plot(
            np.array([0,end_time])-move_time,
            [hold_thresh,hold_thresh],
            '--k'
    )
    if type(trial['idx_rtHoldTimes']) is np.ndarray:
        go_cue_idx = trial['idx_goCueTime']
        for event_idx in trial['idx_rtHoldTimes']:
            axs[0].plot(
                np.repeat((event_idx-go_cue_idx)*bin_size,2),
                [0,max_speed],
                '--r',
            )

    axs[0].set_ylim(0,max_speed)
    axs[0].set_ylabel('Hand Speed')
    axs[0].set_xlabel('Time (s)')
    
    axs[1].hist(
        hold_sig,
        bins=np.linspace(0,max_speed,30),
        density=True,
        orientation='horizontal',
        color='k',
    )
    axs[2].hist(
        move_sig,
        bins=np.linspace(0,max_speed,30),
        density=True,
        orientation='horizontal',
        color='r',
    )
    for ax in axs:
        sns.despine(ax=ax,trim=True)

    # plot out hand position
    pos_ax.plot([-move_time,end_time-move_time],[0,0],'-k')
    pos_ax.plot(
        np.arange(0,move_time,bin_size)-move_time,
        trial['rel_hand_pos'][hold_slice_fun(trial),0],
        c='k',
    )
    pos_ax.plot(
        bin_size*np.arange(move_sig.shape[0]),
        trial['rel_hand_pos'][move_slice_fun(trial),0],
        c='r',
    )
    pos_ax.set_ylim(-35,35)
    pos_ax.set_ylabel('Hand position')
    pos_ax.set_title(f'Trial {trial.name} ({trial["task"]})')
    sns.despine(ax=pos_ax,trim=True)


# def find_move_thresh(trial,hold_slice_fun=None):
#     if hold_slice_fun is None:
#         hold_slice_fun = pyaldata.generate_epoch_fun(
#                 start_point_name='idx_goCueTime',
#                 rel_start = -0.4/trial['bin_size'],
#                 rel_end = 0,
#                 )
# 
#     hold_sig
#     thresh = np.percentile(abs(hold_sig),99,axis=0)
# 
#     return thresh
