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


def visualization(hor, ver, mask_prediction, mask_gt):
    array1 = mask_prediction
    array2 = mask_gt
    array3 = hor[:1,:,:]
    array4 = ver[:1,:,:]

    # indices = np.argwhere(array2[0,:,:] == 1)
    # if indices.shape != (0,2):
    #     middle_index = np.median(indices, axis=0).astype(int)[::-1]
    #     dots_array.append((middle_index[0],middle_index[1]))
    # else:
    #     dots_array.append((0,0))
    # dots = [middle_index]
    # colors = ['r']

    # dots_array.append((c_x,c_y))

    fig, axs = plt.subplots(1, 4)

    # Plot the first image in the left subplot
    axs[0].imshow(array3.squeeze())
    axs[0].axis('off')

    # Plot the second image in the right subplot
    axs[1].imshow(array4.squeeze())
    axs[1].axis('off')

    # Plot the third image in the right subplot
    axs[2].imshow(array1.squeeze())
    axs[2].axis('off')

    # Plot the fourth image in the right subplot
    axs[3].imshow(array2.squeeze())
    axs[3].axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1)
    plt.title("IoU: "+str(np.round(1,3)))

    # Save the figure
    # address = './visualization/'+str(filename.item())+".png"
    # plt.savefig(address)

    # Show the figure
    plt.show()

    plt.close()