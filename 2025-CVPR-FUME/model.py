import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=300, output_dim=20, mid_num=4096, layer_num=3):
        super(ImgNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, mid_num))
        for _ in range(layer_num - 1):
            self.layers.append(nn.Linear(mid_num, mid_num))
        self.layers.append(nn.Linear(mid_num, output_dim, bias=False))
        
    def forward(self, x):
        x = x.to(torch.float)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / norm


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=28*28, output_dim=20, mid_num=4096, layer_num=3):
        super(TextNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, mid_num))
        for _ in range(layer_num - 1):
            self.layers.append(nn.Linear(mid_num, mid_num))
        self.layers.append(nn.Linear(mid_num, output_dim, bias=False))
        
    def forward(self, x):
        x = x.to(torch.float)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / norm

class FUME(nn.Module):
    def __init__(self, device, img_input_dim=4096, text_input_dim=1024, output_dim =1024, num_class=10, layer_num=3):
        super(FUME, self).__init__()
        self.img_net = ImgNN(img_input_dim, output_dim =output_dim,layer_num=layer_num)
        self.text_net = TextNN(text_input_dim, output_dim =output_dim, layer_num=layer_num)
        
        W = torch.Tensor(output_dim, output_dim)
        self.W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: num_class].to(device)
        norm = torch.norm(self.W, p=2, dim=1, keepdim=True)
        self.W = self.W / norm

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)

        W = self.W / torch.norm(self.W, p=2, dim=1, keepdim=True) # Change p to 3, 4, or 5 may result in better performance!
        view1_predict = view1_feature.view([view1_feature.shape[0], -1]).mm(W)
        view2_predict = view2_feature.view([view2_feature.shape[0], -1]).mm(W)

        view1_membershipDegree = torch.relu(view1_predict)
        view2_membershipDegree = torch.relu(view2_predict)

        view1_cred = self.get_test_category_credibility(view1_membershipDegree)
        view2_cred = self.get_test_category_credibility(view2_membershipDegree)
        view1_uncertainty = self.get_fuzzyUncertainty(view1_cred)
        view2_uncertainty = self.get_fuzzyUncertainty(view2_cred)

        ret = dict()
        ret['view1_feature'] = view1_feature
        ret['view2_feature'] = view2_feature
        ret['view1_membershipDegree'] = view1_membershipDegree
        ret['view2_membershipDegree'] = view2_membershipDegree
        ret['view1_uncertainty'] = view1_uncertainty
        ret['view2_uncertainty'] = view2_uncertainty

        return ret

    def get_test_category_credibility(self, membershipDegree): 
        top2MembershipDegree = torch.topk(membershipDegree, k=2, dim=1, largest=True, sorted=True)[0]
        category_credibility = membershipDegree - top2MembershipDegree[:, 0].view([-1, 1]).detach() + 1
        category_credibility += (category_credibility == 1).float() * (top2MembershipDegree[:, 0] 
                                                                       - top2MembershipDegree[:, 1]).reshape([-1, 1]).detach()  
        
        return category_credibility/2

    def get_fuzzyUncertainty(self, category_credibility):
        nonzero_indices = torch.nonzero(category_credibility)
        class_num = category_credibility.shape[1]
        e = 0.0000001
        if len(nonzero_indices) > 1:
            H = torch.sum((-category_credibility*torch.log(category_credibility+e) 
                           - (1-category_credibility)*torch.log(1-category_credibility+e)), dim=1, keepdim=True)
            uncertainty = H / (class_num * torch.log(torch.tensor(2)))
        else:
            uncertainty = torch.tensor(0).unsqueeze(0)
            
        return uncertainty