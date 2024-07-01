import torch
import os
import numpy as np

from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import utils.img_utils as utils


def make_tuning_step(model, optimizer, target_representation, should_reconstruct_from_content, content_feature_maps_index, style_feature_maps_indices): 

    def tuning_step(optimizing_img): 
        optimizer.zero_grad()
        set_of_feature_maps = model(optimizing_img)
        if should_reconstruct_from_content: 
            current_representation = set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
        else: 
            current_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices]

        loss = 0.0 
        if should_reconstruct_from_content: 
            loss = torch.nn.MSELoss(reduction='mean')(target_representation, current_representation)
        else: 
            for gram_target, gram_current in zip(target_representation, current_representation):
                loss += (1/len(target_representation)) * torch.nn.MSELoss(reduction='sum')(gram_target[0], gram_current[0])

        loss.backward()
        optimizer.step()

        return loss.item(), current_representation
    
    return tuning_step

def reconstruction_of_image(configuration_obj): 
    should_reconstruct_from_content = configuration_obj('should_reconstruct_content')
    should_visualize_representation = configuration_obj['should_visualize_representation']
    dump_path = os.path.join(configuration_obj['output_img_dir'], ('c' if should_reconstruct_from_content else 's') + '_reconstruction_' + configuration_obj['optimizer'])
    dump_path = os.path.join(dump_path, os.path.basename(configuration_obj['content_img_name']).split('.')[0] if should_reconstruct_from_content else os.path.basename(configuration_obj['style_img_name']).split('.')[0])
    os.makedirs(dump_path, exist_ok=True)

    content_image_path = os.path.join(configuration_obj['content_images_dir'], configuration_obj['content_img_name'])
    style_image_path = os.path.join(configuration_obj['style_images_dir'], configuration_obj['style_img_name'])
    img_path = content_image_path if should_reconstruct_from_content else style_image_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = utils.prepare_image(img_path, configuration_obj['height'], device)

    gaussian_noise_img = np.random.normal(loc=0, scale=90., size=img.shape)
    white_noise_img = np.random.uniform(-90., 90., img.shape).astype(np.float32)


    initial_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    optimizing_img = Variable(initial_img, requires_grad = True)

    number_of_iterations = { 'adam': 2500, 'lbfgs': 300}
    neural_network, content_fms_index_names, style_fms_indicies_names = utils.prepare_model(configuration_obj['model'], device)

    set_of_feature_maps = neural_network(img)

    if should_reconstruct_from_content: 
        target_content_representation = set_of_feature_maps[content_fms_index_names[0]].squeeze(axis=0)
        if should_visualize_representation:
            num_of_feature_maps = target_content_representation.size()[0]
            print(f'Number of feature maps: {num_of_feature_maps}')
            for i in range(num_of_feature_maps):
                feature_map = target_content_representation[i].to('cpu').numpy()
                feature_map = np.uint8(utils.get_uint8_range(feature_map))
                plt.imshow(feature_map)
                plt.title(f'Feature map {i+1}/{num_of_feature_maps} from layer {content_fms_index_names[1]} (model={configuration_obj["model"]}) for {configuration_obj["content_img_name"]} image.')
                plt.show()
                filename = f'fm_{configuration_obj["model"]}_{content_fms_index_names[1]}_{str(i).zfill(configuration_obj["img_format"][0])}{configuration_obj["img_format"][1]}'
                utils.save_image(feature_map, os.path.join(dump_path, filename))


    else: 
        target_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_fms_indicies_names[0]]
        if should_visualize_representation: 
            number_of_gram_matrices = len(target_style_representation)
            print(f"Number of Gram matrices : {number_of_gram_matrices}")

            for i in range(number_of_gram_matrices):
                Gram_matrix = target_style_representation[i].squeeze(axis=0).to('cpu').numpy()
                Gram_matrix = np.uṭint8(utils.get_uint8_range(Gram_matrix))
                plt.imshow(Gram_matrix)
                plt.title(f'Gram matrix from layer {style_fms_indicies_names[1][i]} (model={configuration_obj["model"]}) for {configuration_obj["style_img_name"]} image.')
                plt.show()
                filename = f'gram_{configuration_obj["model"]}_{style_fms_indicies_names[1][i]}_{str(i).zfill(configuration_obj["img_format"][0])}{configuration_obj["img_format"][1]}'
                utils.save_image(Gram_matrix, os.path.join(dump_path, filename))

    if configuration_obj['optimizer'] == 'adam': 
        optimizer = Adam((optimizing_img, ))

        target_representation = target_content_representation if should_reconstruct_from_content else target_style_representation
        tuning_step = make_tuning_step(neural_network, optimizer, target_representation, should_reconstruct_from_content, content_fms_index_names[0], style_fms_indicies_names[0])
        for it in range(number_of_iterations[configuration_obj['optimizer']]): 
            loss, _ = tuning_step(optimizing_img)
            with torch.no_grad(): 
                 print(f'Iteration: {it}, current {"content": if should_reconstruct_content else "style"} loss={loss : 10.8f}')
    elif configuration_obj['optimizer'] == 'lbfgs': 
        count = 0 

        def closure(): 
            nonlocal count
            optimizer.zero_grad()
            loss = 0.0 

            if should_reconstruct_from_content: 
                loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, neural_network(optimizing_img)[content_fms_index_names[0]].squeeze(axis=0))
            else:
                current_set_of_feature_maps = neural_network(optimizing_img)
                current_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(current_set_of_feature_maps) if i in style_fms_indicies_names[0]]
                for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                    loss += (1 / len(target_style_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
            loss.backward()
            with torch.no_grad():
                print(f'Iteration: {count}, current {"content" if should_reconstruct_from_content else "style"} loss={loss.item()}')
                count += 1

            return loss
        
        optimizer = torch.optim.LBFGS((optimizing_img,), max_iter = number_of_iterations[configuration_obj['optimizer']], line_search_fn='storng_wolfe')
        optimizer.step(closure)

    return dump_path


if __name__ == '__main__': 

    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content_images')
    style_images_dir = os.path.join(default_resource_dir, 'style_images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')

    parser = argparse.ArgumentParser()
    parser.add_argument("--should_reconstruct_content", type="bool", )
    parser.add_argument("--should_visualize_representation", type=bool, help="visualize feature maps or Gram matrices", default=False)

    parser.add_argument("--content_img_name", type=str, help="content image name", default='lion.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='ben_giles.jpg')
    parser.add_argument("--height", type=int, help="width of content and style images (-1 keep original)", default=500)

    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=1)
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--reconstruct_script", type=str, help='dummy param - used in saving func', default=True)
    args = parser.parse_args()

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    results_path = reconstruction_of_image(optimization_config)