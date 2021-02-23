import torch
import yaml
from easydict import EasyDict as edict
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
from torch.autograd import Variable
import PIL
import torchvision.utils as vutils
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
T = torch.cuda

with open('src/cfg/pororo_s1.yml', 'r') as f:
        cfg = edict(yaml.load(f))

cfg.TRAIN.FLAG = False
cfg.CUDA = True

label_array = np.zeros(1756)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION * cfg.VIDEO_LEN
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

class StoryGAN(nn.Module):
    def __init__(self, video_len):
        super(StoryGAN, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.motion_dim = cfg.TEXT.DIMENSION + cfg.LABEL_NUM
        self.content_dim = cfg.GAN.CONDITION_DIM # encoded text dim
        self.noise_dim = cfg.Z_DIM  # noise
        self.recurrent = nn.GRUCell(self.noise_dim + self.motion_dim, self.motion_dim)
        self.mocornn = nn.GRUCell(self.motion_dim, self.content_dim)
        self.video_len = video_len
        self.n_channels = 3
        self.filter_num = 3
        self.filter_size = 21
        self.image_size = 124
        self.out_num = 1
        self.define_module()

    def define_module(self):
        from src.layers import DynamicFilterLayer1D as DynamicFilterLayer
        ninput = self.motion_dim + self.content_dim + self.image_size
        ngf = self.gf_dim

        self.ca_net = CA_NET()
        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim))

        self.dfn_layer = DynamicFilterLayer(self.filter_size, 
            pad = self.filter_size//2)

    def get_iteration_input(self, motion_input):
        num_samples = motion_input.shape[0]
        noise = T.FloatTensor(num_samples, self.noise_dim).normal_(0,1)
        return torch.cat((noise, motion_input), dim = 1)

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.motion_dim).normal_(0, 1))

    def sample_z_motion(self, motion_input, video_len=None):
        video_len = video_len if video_len is not None else self.video_len
        num_samples = motion_input.shape[0]
        h_t = [self.m_net(self.get_gru_initial_state(num_samples))]
        
        for frame_num in range(video_len):
            if len(motion_input.shape) == 2:
                e_t = self.get_iteration_input(motion_input)
            else:
                e_t = self.get_iteration_input(motion_input[:,frame_num,:])
            h_t.append(self.recurrent(e_t, h_t[-1]))
        z_m_t = [h_k.view(-1, 1, self.motion_dim) for h_k in h_t]
        z_motion = torch.cat(z_m_t[1:], dim=1).view(-1, self.motion_dim)
        return z_motion

    def motion_content_rnn(self, motion_input, content_input):
        video_len = 1 if len(motion_input.shape) == 2 else self.video_len
        h_t = [self.c_net(content_input)]
        if len(motion_input.shape) == 2:
            motion_input = motion_input.unsqueeze(1)
        for frame_num in range(video_len):
            h_t.append(self.mocornn(motion_input[:,frame_num, :], h_t[-1]))
        
        c_m_t = [h_k.view(-1, 1, self.content_dim) for h_k in h_t]
        mocornn_co = torch.cat(c_m_t[1:], dim=1).view(-1, self.content_dim)
        return mocornn_co

    def sample_videos(self, motion_input, content_input):  
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        r_code, r_mu, r_logvar = self.ca_net((content_input))
        c_code = r_code.repeat(self.video_len, 1).view(-1, r_code.shape[1])
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])
        c_logvar = r_logvar.repeat(self.video_len, 1).view(-1, r_logvar.shape[1])

        crnn_code = self.motion_content_rnn(motion_input, r_code)
        
        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp #self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code, self.video_len)

        # one
        zmc_code = torch.cat((zm_code, c_mu), dim = 1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2]))
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter])
        zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        #combine
        zmc_all = self.fc(zmc_all)
        zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(zmc_all)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        h = self.img(h_code)
        fake_video = h.view(int(h.size(0)) // int(self.video_len), int(self.video_len), int(self.n_channels), int(h.size(3)), int(h.size(3)))
        fake_video = fake_video.permute(0, 2, 1, 3, 4)
        return None, fake_video,  m_mu, m_logvar, r_mu, r_logvar

def inference(desc):
    if os.path.exists('src/temp'):
        pass
    else:
        os.mkdir('src/temp')
    model = StoryGAN(cfg.VIDEO_LEN)
    model.load_state_dict(torch.load('models/storyGAN_model/storyGAN.pth'))
    model.eval()
    model.to('cuda:0')

    labels_dict = {
        "barney" : 78,
        "fred" : 1490,
        "wilma" : 1753,
        "betty" : 1011,
    }

    for word in desc.lower().split(' '):
        if word in labels_dict:
            label_array[labels_dict[word]] = 1

    # label_array[78] = 1 #Barney
    # label_array[1490] = 1 #Fred
    # label_array[1753] = 1 #Wilma
    # label_array[1011] = 1 #Betty
    # label_array[895] = 1 #Gazoo
    # label_array[67] = 1 #Mr. slate
    # label_array[103] = 1 #Pearl
    # label_array[25] = 1 #Blue Dino
    # label_array[502] = 1 #dinosaur bird

    desc_enc = encoder.encode(desc,show_progress_bar=False)

    desc_tensor = Variable(torch.Tensor(desc_enc))
    label_tensor = Variable(torch.Tensor(label_array))

    desc_temp = np.zeros([1,cfg.VIDEO_LEN,768])
    for i in range(cfg.VIDEO_LEN):
        desc_temp[0][i] = (desc_tensor)
    desc_tensor = torch.Tensor(desc_temp)
    label_temp = np.zeros([1,cfg.VIDEO_LEN,1756])
    for i in range(cfg.VIDEO_LEN):
        label_temp[0][i] = (label_tensor)
    label_tensor = torch.Tensor(label_temp)

    desc_tensor = Variable(desc_tensor).to(device)
    label_tensor = Variable(label_tensor).to(device)
    motion_input = torch.cat((desc_tensor.to(device),label_tensor.to(device)),2).to(device)
    content_input = desc_tensor.to(device)

    _, images, _,_,_,_ = model.sample_videos(motion_input, content_input)

    def images_to_numpy(tensor):
        generated = tensor.data.cpu().numpy().transpose(1,2,0)
        generated[generated < -1] = -1
        generated[generated > 1] = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype('uint8')


    video_len = cfg.VIDEO_LEN
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(torch.transpose(images[i], 0,1), video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)

    images = []
    sizes = [(2,66),(67,131),(134,198),(200,264),(266,330)]
    for i in range(0,5):
        PIL.Image.fromarray(all_images).crop((sizes[i][0],2,sizes[i][1],64)).save('src/temp/'+str(i+1)+'.png')

    import imageio
    images = []
    for i in range(0,5):
        images.append(imageio.imread('src/temp/'+str(i+1)+'.png'))
    imageio.mimsave('src/temp/out.gif', images)
    f = open('src/temp/out.gif','r')
    return f
