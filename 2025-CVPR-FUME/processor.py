from __future__ import print_function
from __future__ import division
import torch
import copy
from evaluate import *
import numpy as np

def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha):
    def get_train_category_credibility(predict,labels):
        top1Possibility = (predict*(1-labels)).max(1)[0].reshape([-1,1])
        labelPossibility = (predict*labels).max(1)[0].reshape([-1,1])
        neccessity = (1-labelPossibility)*(1-labels) + (1-top1Possibility)*labels
        r = (predict + neccessity)/2
        return r
    def consistency_Learning(fea, tau=1.):
        n_view = 2
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())

        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss1 = -(diag1 / sim.sum(1)).log().mean()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss2 = -(diag2 / sim.sum(1)).log().mean()
        return loss1 + loss2

    view1_cred = get_train_category_credibility(view1_predict,labels_1)
    view2_cred = get_train_category_credibility(view2_predict,labels_2)
    loss_fml = ((view1_cred-labels_1.float())**2).sum(1).sqrt().mean() + ((view2_cred-labels_2.float())**2).sum(1).sqrt().mean()
    
    loss_cl = consistency_Learning([view1_feature,view2_feature])
    loss_total = loss_fml + alpha * loss_cl

    return loss_total

def show_results_under_uncertainty(img_uncertainty, txt_uncertainty, t_imgs, t_txts, t_labels, phase):
    threshold = [0.5]

    uncertainty = img_uncertainty @ txt_uncertainty.T
    img_retrieval_uncertainty = copy.deepcopy(uncertainty)
    txt_retrieval_uncertainty = copy.deepcopy(uncertainty)

    for i in range(img_uncertainty.shape[0]):
        for j in range(txt_uncertainty.shape[0]):
            img_retrieval_uncertainty[i][j] = 1-(1-img_uncertainty[i])*(1-txt_uncertainty[j])

    for i in range(txt_uncertainty.shape[0]):
        for j in range(img_uncertainty.shape[0]):
            txt_retrieval_uncertainty[i][j] = 1-(1-txt_uncertainty[i])*(1-img_uncertainty[j])

    for t in threshold:
        img2txt_ = fx_calc_map_label_withUncertainty(t_imgs, t_txts, t_labels, img_retrieval_uncertainty, 0, t)
        txt2img_ = fx_calc_map_label_withUncertainty(t_txts, t_imgs, t_labels, txt_retrieval_uncertainty, 0, t)
        if not np.isnan(img2txt_) and not np.isnan(txt2img_):
            print('**** {}ing Uncertainty Threshold={} mAP@all ==> Img2Txt: {:.4f}  Txt2Img: {:.4f}, Aver: {:.4f}'.format(phase, t, img2txt_, txt2img_, (img2txt_+txt2img_)/2 ))

    return img2txt_, txt2img_, img_retrieval_uncertainty, txt_retrieval_uncertainty

def train_val_test_model(model, data_loaders, optimizer, alpha, num_epochs=500, device="cpu", folder_path=None):

    results = dict()
    results['best_val_Aver_mAP'] = 0.0
    results['corresponding_test_Aver_mAP'] = 0.0
    results['corresponding_test_I2T_mAP'] = 0.0
    results['corresponding_test_T2I_mAP'] = 0.0
    results['corresponding_test_Aver_mAP_Threshold=0.5'] = 0.0
    results['corresponding_test_I2T_mAP_Threshold=0.5'] = 0.0
    results['corresponding_test_T2I_mAP_Threshold=0.5'] = 0.0


    for epoch in range(num_epochs):
        save_flag = False
        train_loss = 0
        
        print('-' * 80)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for imgs, txts, labels in data_loaders[phase]:
                imgs = imgs.to(device)
                txts = txts.to(device)
                labels = labels.to(device)
                if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
                    print("Data contains Nan.")

                # optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()

                    ret = model(imgs, txts)

                    view1_feature, view2_feature = ret['view1_feature'], ret['view2_feature']
                    view1_predict, view2_predict = ret['view1_membershipDegree'], ret['view2_membershipDegree']
                    
                    loss = calc_loss(view1_feature, view2_feature, view1_predict,
                                     view2_predict, labels, labels, alpha)
                    
                    train_loss += loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if np.isnan(epoch_loss):
                break                

            if phase == 'test' or phase == 'val':
                t_imgs, t_txts, t_labels, t_imgs_uncer, t_txts_uncer = [], [], [], [], []
                with torch.no_grad():
                    for imgs, txts, labels in data_loaders[phase]:
                        imgs = imgs.to(device)
                        txts = txts.to(device)
                        labels = labels.to(device)

                        ret = model(imgs, txts)
                        t_view1_feature = ret['view1_feature']
                        t_view2_feature = ret['view2_feature']
                        t_view1_uncer = ret['view1_uncertainty']
                        t_view2_uncer = ret['view2_uncertainty']

                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())

                        t_labels.append(labels.cpu().numpy())
                        t_imgs_uncer.append(t_view1_uncer.cpu().numpy())
                        t_txts_uncer.append(t_view2_uncer.cpu().numpy())

                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)

                t_imgs_uncer = np.concatenate(t_imgs_uncer)
                t_txts_uncer = np.concatenate(t_txts_uncer) 
                t_labels = np.concatenate(t_labels).argmax(1)

                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels, 0)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels, 0)

                print('{}ing  Loss: {:.4f} mAP@all ==> Img2Txt: {:.4f}  Txt2Img: {:.4f}, Aver: {:.4f}'.format(phase, epoch_loss, img2text, txt2img, (img2text+txt2img)/2 ))

                if phase == 'test' and save_flag == True:
                    results['corresponding_test_I2T_mAP'] = img2text 
                    results['corresponding_test_T2I_mAP'] = txt2img 
                    results['corresponding_test_Aver_mAP'] = (img2text + txt2img) / 2.

                    img2txt_t, txt2img_t, img2txt_u, txt2img_u = show_results_under_uncertainty(t_imgs_uncer, t_txts_uncer, t_imgs, t_txts, t_labels, phase)

                    results['corresponding_test_I2T_mAP_Threshold=0.5'] = img2txt_t
                    results['corresponding_test_T2I_mAP_Threshold=0.5'] = txt2img_t
                    results['corresponding_test_Aver_mAP_Threshold=0.5'] = (img2txt_t + txt2img_t) / 2.

                    np.save(folder_path+'/I_feat.npy', t_imgs)
                    np.save(folder_path+'/T_feat.npy', t_txts)
                    np.save(folder_path+'/I2T_uncer.npy', img2txt_u)
                    np.save(folder_path+'/T2I_uncer.npy', txt2img_u)
                    
                    print('feature and uncertainty on the testing set are saved.')

                if phase == 'test':
                    print('best valing Aver = {:.4f} ==> corresponding testing Img2Txt: {:.4f}, Txt2Img: {:.4f}, Aver: {:.4f}'.format(\
                        results['best_val_Aver_mAP'], \
                            results['corresponding_test_I2T_mAP'],\
                                results['corresponding_test_T2I_mAP'], \
                                results['corresponding_test_Aver_mAP']))
 

            if phase == 'val' and (img2text + txt2img) / 2. > results['best_val_Aver_mAP']:
                results['best_val_Aver_mAP'] = (img2text + txt2img) / 2.
                save_flag = True

        if np.isnan(epoch_loss):
            print("loss is nan!")    
            break

    return results
