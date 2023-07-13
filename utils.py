import torch


'''gamma selection'''
def gam_logit_select(points, pred, label, rho=0.5):
    #points shape: (batch_size, 1, 256, 256)
    eps = 1e-5
    '''get median position index'''
    slice_points = torch.mean(points.reshape((points.shape[0],-1)), axis=-1) #(16, 224x224)
    med_idx = torch.argsort(slice_points)[len(slice_points)//2] #median index
    #print(pred.shape, med_idx)
    '''compute pos and neg'''
    logit = torch.sigmoid(pred[med_idx]) #compute logit for median
    '''compute positive and negative'''
    pos = (rho-1)/(torch.log(logit)+eps)
    neg = (1-rho)/(torch.log(1+torch.exp(pred[med_idx]))+eps)
    
    #gamma = torch.mean(label[med_idx]*pos+(1-label[med_idx])*neg) #for segmantation
    gamma = label[med_idx]*pos+(1-label[med_idx])*neg
    #gamma = weights*label[med_idx]*pos+(1-weights)*(1-label[med_idx])*neg
    gam_clip = torch.clamp(gamma, 1e-4, 1).data.cpu().numpy()
    return gam_clip

def gamma_logit_loss(pred, label, gamma=1e-4): #1e-4
    '''
    gamma selection: positive number, min: 1e-20
    '''
    pos = torch.sigmoid((1+gamma)*pred) #gamma logistic layer
    neg = 1 - pos
    power = gamma/(1+gamma)
    score = label*((1-(pos**power))/gamma) + (1-label)*((1-(neg**power))/gamma)
    
    #print('mislabels: \n', score[:10])
    #print('normal labels: \n', score[-10:])
    '''gamma selection for this batch, used for computing mean gamma for next FL round'''
    gam_update = gam_logit_select(score, pred, label)
    #gam_update = 1e-4
    gamma_loss = torch.mean(score)
    return gamma_loss, gam_update


def ce_loss(pred, label):
    score = label*torch.log(pred)+(1-label)*torch.log(1-pred)
    ce = -torch.mean(score)
    return ce