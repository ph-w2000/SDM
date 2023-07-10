import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as patches
import numpy as np

def visualize_attention(attention_map):
    # Group the keypoints by taking the mean along the keypoints dimension
    grouped_attention_map = np.mean(attention_map, axis=0)
    # Plot the grouped attention map as a heatmap
    plt.imshow(grouped_attention_map, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Grouped Attention Map')
    plt.show()


def visualization(hor, ver, mask_prediction, mask_gt, bone_prediction, bone_gt, hor_box, ver_box, filename, score):
    array1 = mask_prediction
    array2 = mask_gt
    array3 = hor[:1,:,:]
    array4 = ver[:1,:,:]
    array5 = bone_prediction
    array6 = bone_gt
    array7 = hor_box
    array8 = ver_box

    # indices = np.argwhere(array2[0,:,:] == 1)
    # if indices.shape != (0,2):
    #     middle_index = np.median(indices, axis=0).astype(int)[::-1]
    #     dots_array.append((middle_index[0],middle_index[1]))
    # else:
    #     dots_array.append((0,0))
    # dots = [middle_index]
    # colors = ['r']

    hboxes = array7
    h_x = hboxes[0,0]
    h_y = hboxes[0,1]
    h_width = hboxes[0,2] - h_x
    h_height = hboxes[0,3] - h_y
    vboxes = array8
    v_x = vboxes[0,0]
    v_y = vboxes[0,1]
    v_width = vboxes[0,2] - v_x
    v_height = vboxes[0,3] - v_y

    c_x = h_x + h_width/2
    c_y = h_y + h_height/2

    # dots_array.append((c_x,c_y))

    fig, axs = plt.subplots(1, 4)

    # Plot the first image in the left subplot
    axs[0].imshow(array3.squeeze())
    axs[0].axis('off')
    rect = patches.Rectangle((h_x, h_y), h_width, h_height, linewidth=1, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    rect = patches.Rectangle((c_x, c_y), h_width, h_height, linewidth=1, edgecolor='b', facecolor='none')
    axs[0].add_patch(rect)

    # Plot the second image in the right subplot
    axs[1].imshow(array4.squeeze())
    axs[1].axis('off')
    rect = patches.Rectangle((v_x, v_y), v_width, v_height, linewidth=1, edgecolor='r', facecolor='none')
    axs[1].add_patch(rect)

    # Plot the third image in the right subplot
    axs[2].imshow(array1.squeeze())
    axs[2].axis('off')
    x = array5[0,:, 0]
    y = array5[0,:, 1]
    axs[2].scatter(x, y, color='red', marker='o',s=1)

    # Plot the fourth image in the right subplot
    axs[3].imshow(array2.squeeze())
    axs[3].axis('off')
    x = array6[0,:, 0]
    y = array6[0,:, 1]
    axs[3].scatter(x, y, color='red', marker='o',s=1)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1)
    plt.title("IoU: "+str(np.round(score,3)))

    # Save the figure
    # address = './visualization/'+str(filename.item())+".png"
    # plt.savefig(address)

    # Show the figure
    plt.show()

    plt.close()