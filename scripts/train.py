from tqdm import tqdm
import numpy as np
from PIL import Image
import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from .prepare_dataset import get_mnist_batch_color

def train(args, generator, discriminator, g_optimizer, d_optimizer, epochs, mnist_color = False):

  init_size = args["init_size"]
  max_size = args["max_size"]
  dataset_size = args["dataset_size"]
  z_dim = args["z_dim"]
  n_critic = args["n_critic"]
  gen_sample = args["gen_sample"] 
  store_generator_at_iteration = args["store_generator_at_iteration"]

  step = int(math.log2(init_size)) - 2 # 0 = 4, 1 = 8, 2 = 16
  resolution = 4 * 2 ** step
  loader = trainloader[0]

  data_loader = iter(loader)

  phase = epochs * dataset_size
  no_of_iterations = epochs * dataset_size
  pbar = tqdm(range(no_of_iterations))

  #discriminator training
  requires_grad(generator, False)
  requires_grad(discriminator, True)

  disc_loss_val = 0
  gen_loss_val = 0
  grad_loss_val = 0

  alpha = 0
  used_sample = 0
  n_iter_resolution = 0

  max_step = int(math.log2(max_size)) - 2
  final_progress = False

  #training,
  for i in pbar:
      discriminator.zero_grad() #zero the gradients, 

      alpha = min(1, 1 / phase * (used_sample + 1))

      if (resolution == init_size) or final_progress:
          alpha = 1

      if used_sample >= 2*phase:
          used_sample = 0
          n_iter_resolution = 0
          step += 1

          if step > max_step:
              step = max_step
              final_progress = True
              ckpt_step = step + 1

          else:
              alpha = 0
              ckpt_step = step

          resolution = 4 * 2 ** step
          
          loader = trainloader[step]

          data_loader = iter(loader)

          print(f'moving to step {step} and resolution {resolution}')

          #save the networks states. 
          torch.save(
              {
                  'generator': generator.state_dict(),
                  'discriminator': discriminator.state_dict(),
                  'g_optimizer': g_optimizer.state_dict(),
                  'd_optimizer': d_optimizer.state_dict(),
              },
              f'checkpoint/train_step-{ckpt_step}.model',
          )

      #get the real_image and fake_image for data_loader
      try:
          real_image, real_labels = next(data_loader)

      except (OSError, StopIteration):
          data_loader = iter(loader)
          real_image, real_labels = next(data_loader)
          
      
      if mnist_color:
            real_image = get_mnist_batch_color(real_image)
            real_image = torch.tensor(real_image, dtype=torch.float)

      used_sample += real_image.shape[0]
      n_iter_resolution += 1 

      b_size = real_image.size(0)
      real_image = real_image.cuda()

      #wgan-gp for discriminator, first value
      real_predict = discriminator(real_image, step=step, alpha=alpha)
      real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
      (-real_predict).backward()

      #generator inputs for z
      gen_in1, gen_in2 = torch.randn(2, b_size, z_dim, device='cuda').chunk(
                  2, 0
      )
      gen_in1 = gen_in1.squeeze(0)
      gen_in2 = gen_in2.squeeze(0)

      #wgan-gp for discriminator, second value
      fake_image = generator(gen_in1, batch_size=BATCH_SIZE, step=step, alpha=alpha)
      fake_predict = discriminator(fake_image, step=step, alpha=alpha)

      fake_predict = fake_predict.mean()
      fake_predict.backward()

      #GP calculation for third value
      eps = torch.rand(b_size, 1, 1, 1).cuda()
      x_hat = eps * real_image.data + (1 - eps) * fake_image.data
      x_hat.requires_grad = True
      hat_predict = discriminator(x_hat, step=step, alpha=alpha)
      grad_x_hat = grad(
          outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
      )[0]
      grad_penalty = (
          (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
      ).mean()
      grad_penalty = 10 * grad_penalty
      grad_penalty.backward()


      if i%10 == 0:
          grad_loss_val = grad_penalty.item()
          disc_loss_val = (-real_predict + fake_predict).item()

      d_optimizer.step()

      if (i + 1) % n_critic == 0:
          generator.zero_grad()

          requires_grad(generator, True)
          requires_grad(discriminator, False)

          fake_image = generator(gen_in2, batch_size=BATCH_SIZE, step=step, alpha=alpha)

          predict = discriminator(fake_image, step=step, alpha=alpha)

          loss = -predict.mean()

          if i%10 == 0:
              gen_loss_val = loss.item()

          loss.backward()
          g_optimizer.step()

          requires_grad(generator, False)
          requires_grad(discriminator, True)

      #for tensorboard. 
      writer.add_scalars(f"{resolution}_Loss", 
                        {
                            "disc_loss_val": disc_loss_val, 
                            "gen_loss_val": gen_loss_val,
                            "grad_loss_val": grad_loss_val 
                        }, 
                        n_iter_resolution)
      

      if (i + 1) % 100 == 0:
          images = []

          gen_i, gen_j = gen_sample.get(resolution, (8, 8))

          with torch.no_grad():
              for _ in range(gen_i):
                  images.append(
                      generator(
                          torch.randn(gen_j, z_dim).cuda(), batch_size=BATCH_SIZE, step=step, alpha=alpha
                      ).data.cpu()
                  )

          utils.save_image(
              torch.cat(images, 0),
              f'sample/{str(i + 1).zfill(6)}.png',
              nrow=gen_i,
              normalize=True,
              range=(-1, 1),
          )

      if (i + 1) % store_generator_at_iteration == 0:
          torch.save(
              generator.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
          )

      state_msg = (
          f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
          f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}; step: {step:.5f};' 
          f' n_iter_resolution: {n_iter_resolution:.3f}'
      )
      
      pbar.set_description(state_msg)
  
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag