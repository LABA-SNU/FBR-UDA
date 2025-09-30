import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm.auto import tqdm
from torchmetrics.classification import F1Score
from torchmetrics.functional.classification import multiclass_confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'uda_losses')))
from coral_loss import CORAL_loss
from ddc_loss import MMD_loss, DDC_loss
from nwd import NuclearWassersteinDiscrepancy


def get_trainer(name, model, device, class_weights = None, lambda_nwd=None):
    if (name == "dann") or (name == "dadann_w_reshape"):
        trainer = DANNTrainer(model, device, class_weights = class_weights)
        return trainer
    elif (name == "dann_wo_reshape") or (name == "dadann") or (name == 'cdan'):
        trainer = DANNTrainer(model, device, class_weights = class_weights, reshape = False)
        return trainer    
    elif name == "dcoral":
        trainer = DCORALTrainer(model, device, class_weights = class_weights)
        return trainer
    elif name == "ddc":
        trainer = DDCTrainer(model, device, class_weights = class_weights)
        return trainer
    elif name == "daln":
        trainer = DALNTrainer(model, device, class_weights = class_weights, lambda_nwd=lambda_nwd)
        return trainer
    elif name == "adamatch":
        encoder = model.encoder
        classifier = model.classifier
        
        hyperparams = {"tau": 0.9, "patience": 20, "mu_max": 1.0}
        trainer = AdamatchTrainer(encoder, classifier, device, hyperparams, class_weights=class_weights)
        return trainer
    elif name == "vanilaresnet":
        trainer = VanilaTrainer(model, device, class_weights = class_weights)
        return trainer
    else:
        raise RuntimeError("model \"{}\" not available".format(name))


class BaseTrainer:
    def __init__(self, model, device, class_weights = None):
        self.model = model
        self.device = device
        self.class_weights = class_weights

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

'''
############################
           DANN
############################
'''

class DANNTrainer(BaseTrainer):
    def __init__(self, model, device, class_weights = None, reshape = True):
        super().__init__(model, device, class_weights)
        
        self.num_classes = self.model.classifier.fc.out_features
        
        loss_class = nn.CrossEntropyLoss(weight=class_weights) 
        self.loss_class = loss_class.to(self.device)
        
        loss_domain = nn.CrossEntropyLoss()
        self.loss_domain = loss_domain.to(self.device)

        self.reshape = reshape

    def train(self, dataloader_source, dataloader_target, optimizer, scheduler, epoch, n_epoch):
        
        self.model.train()
        
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_zip = enumerate(zip(dataloader_source, dataloader_target))

        running_loss = 0
        acc_cls = 0
        loss_class_ = 0
        loss_domain_ = 0
        acc_domain_ = 0
        src_acc_domain_ = 0
        tgt_acc_domain_ = 0
        src_total = 0
        tgt_total = 0
        n_total = 0
        src_preds = []
        src_target = []

        for step, ((images_src, class_src), (images_tgt, _)) in tqdm(data_zip, total=len_dataloader):
            #p = float(step + epoch * len_dataloader) / n_epoch / len_dataloader
            #alpha = 2. / (1. + np.exp(-10 * p)) - 1

            alpha = (epoch+1) / n_epoch
        
            size_src = len(images_src)
            size_tgt = len(images_tgt)
            
            if self.reshape:
                label_src = torch.zeros(size_src*49).long().to(self.device)  # source 0
                label_tgt = torch.ones(size_tgt*49).long().to(self.device)  # target 1
            else:
                label_src = torch.zeros(size_src).long().to(self.device)  # source 0
                label_tgt = torch.ones(size_tgt).long().to(self.device)  # target 1
            
            src_target.extend(class_src.numpy())
            class_src = class_src.to(self.device)
            images_src = images_src.to(self.device)
            images_tgt = images_tgt.to(self.device)
        
            self.model.zero_grad()
        
            src_class_output, src_domain_output = self.model(input_data=images_src, alpha=alpha)
            src_preds.extend(src_class_output.detach().cpu().numpy())
            src_loss_class = self.loss_class(src_class_output, class_src) 
            src_loss_domain = self.loss_domain(src_domain_output, label_src)
        
            _, tgt_domain_output = self.model(input_data=images_tgt, alpha=alpha)
            tgt_loss_domain = self.loss_domain(tgt_domain_output, label_tgt)

            total_loss = src_loss_class + (src_loss_domain + tgt_loss_domain)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            src_pred_cls = src_class_output.data.max(1)[1]
            src_pred_domain = src_domain_output.data.max(1)[1]
            tgt_pred_domain = tgt_domain_output.data.max(1)[1]

            loss_class_ += src_loss_class.item() * images_src.size(0)
            loss_domain_ += src_loss_domain.item() * images_src.size(0) + tgt_loss_domain.item() * images_tgt.size(0)
            acc_cls += src_pred_cls.eq(class_src.data).sum().item()
            src_acc_domain_ += src_pred_domain.eq(label_src.data).sum().item()
            tgt_acc_domain_ += tgt_pred_domain.eq(label_tgt.data).sum().item()

            src_total += size_src
            tgt_total += size_tgt
            n_total += (size_src + size_tgt)
        
        src_loss = loss_class_ / src_total
        domain_loss = loss_domain_ / (src_total + tgt_total)
        src_acc = acc_cls / src_total

        if self.reshape:
            src_domain_acc = src_acc_domain_ / (src_total*49)
            tgt_domain_acc = tgt_acc_domain_ / (tgt_total*49)
            domain_acc = (src_acc_domain_ + tgt_acc_domain_) / (n_total*49)
        else:
            src_domain_acc = src_acc_domain_ / (src_total)
            tgt_domain_acc = tgt_acc_domain_ / (tgt_total)
            domain_acc = (src_acc_domain_ + tgt_acc_domain_) / (n_total)
            

        src_target = torch.tensor(src_target)
        src_preds = torch.tensor(np.array(src_preds))
        
        f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        f1_score = f1(src_preds, src_target).item()

        print(f"Training : Avg Loss = {src_loss:.6f}, Avg Accuracy = {src_acc:.2%}, {acc_cls}/{src_total}, F1 Score = {f1_score:.2%}, Avg Domain Accuracy = {domain_acc:.2%}, src = {src_domain_acc:.2%} / tgt = {tgt_domain_acc:.2%}, Avg Domain Loss = {domain_loss:.6f}")
        
        return self.model, src_loss, f1_score


    def evaluate(self, data_loader):
        self.model.eval()
        
        with torch.no_grad():
            loss_ = 0.0
            acc_ = 0.0
            n_total = 0

            pred = []
            target = []
        
            for images, labels in data_loader:
                images = images.to(self.device)
                target.extend(labels.numpy())
                labels = labels.to(self.device) 
                size = len(labels)
                preds, _ = self.model(images, alpha=0)
                loss_ += self.loss_class(preds, labels).item() * images.size(0)
                pred_cls = preds.data.max(1)[1]
                pred.extend(pred_cls.cpu().numpy())
                acc_ += pred_cls.eq(labels.data).sum().item()
                n_total += size
        
            loss = loss_ / n_total
            acc = acc_ / n_total
            
            target = torch.tensor(target)
            pred = torch.tensor(pred)

            f1 = F1Score(task="multiclass", num_classes=self.num_classes)
            f1_score = f1(pred, target).item()
            
            print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, F1 Score = {f1_score:.2%}")

            return loss, acc, f1_score

'''
############################
           DCORAL
############################
'''

class DCORALTrainer(BaseTrainer):
    def __init__(self, model, device, class_weights = None):
        super().__init__(model, device, class_weights)
        self.num_classes = self.model.fc8.out_features
        
        loss_class = nn.CrossEntropyLoss(weight=self.class_weights) 
        self.loss_class = loss_class.to(self.device)
        
        self.CORAL_loss = CORAL_loss

    def train(self, source_loader, target_loader, optimizer, scheduler, epoch, n_epoch):
        self.model.train()

        lambda_factor = (epoch+1) / n_epoch
        #results = []

        pred = []
        label = []

        len_dataloader = min(len(source_loader), len(target_loader))
        data_zip = enumerate(zip(source_loader, target_loader))

        running_loss = 0
        acc_cls = 0
        loss_class_ = 0
        loss_domain_ = 0
        acc_domain_ = 0
        src_acc_domain_ = 0
        tgt_acc_domain_ = 0
        src_total = 0
        tgt_total = 0
        n_total = 0
        src_preds = []
        src_target = []

        for step, ((source_data, source_label), (target_data, _)) in tqdm(data_zip, total=len_dataloader):

            source_data = source_data.to(self.device)
            source_label = source_label.to(self.device)
            target_data = target_data.to(self.device)

            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)

            optimizer.zero_grad()

            feature_output1, feature_output2, output1, output2 = self.model(source_data, target_data)
            classification_loss = self.loss_class(output1, source_label)
            coral_loss = self.CORAL_loss(feature_output1, feature_output2)
            total_loss = classification_loss + lambda_factor * coral_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            pred_cls = output1.data.max(1)[1]
            pred.extend(pred_cls.cpu().numpy())
            label.extend(source_label.cpu().numpy())

        label = torch.tensor(label)
        pred = torch.tensor(pred)
            
        f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        f1_score = f1(pred, label).item()
        
        return self.model, classification_loss, f1_score

    def evaluate(self, data_loader):
        self.model.eval()
        
        with torch.no_grad():
            loss_ = 0.0
            acc_ = 0.0
            n_total = 0
            
            pred = []
            target = []
        
            for images, labels in data_loader:
                images = images.to(self.device)
                
                target.extend(labels.numpy())
                labels = labels.to(self.device) 
                size = len(labels)
        
                #data, label = Variable(data), Variable(label)
                _, _, preds, _ = self.model(images, images)
                loss_ += self.loss_class(preds, labels).item()
                pred_cls = preds.data.max(1)[1]
                pred.extend(pred_cls.cpu().numpy())

                acc_ += pred_cls.eq(labels.data).sum().item()
                n_total += size

            loss = loss_ / n_total
            acc = acc_ / n_total
            
            target = torch.tensor(target)
            pred = torch.tensor(pred)
                
            f1 = F1Score(task="multiclass", num_classes=self.num_classes)
            f1_score = f1(pred, target).item()
        
            print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, F1 Score = {f1_score:.2%}")

            return loss, acc, f1_score

'''
############################
         DDC(MMD)
############################
'''

class DDCTrainer(BaseTrainer):
    def __init__(self, model, device, class_weights = None):
        super().__init__(model, device, class_weights)
        self.num_classes = self.model.fc8.out_features
        
        loss_class = nn.CrossEntropyLoss(weight=self.class_weights) 
        self.loss_class = loss_class.to(self.device)
        
        self.DDC_loss = DDC_loss

    def train(self, source_loader, target_loader, optimizer, scheduler, epoch, n_epoch):
        self.model.train()

        lambda_factor = (epoch+1) / n_epoch
        #results = []

        pred = []
        label = []

        len_dataloader = min(len(source_loader), len(target_loader))
        data_zip = enumerate(zip(source_loader, target_loader))

        for step, ((source_data, source_label), (target_data, _)) in tqdm(data_zip, total=len_dataloader):

            source_data = source_data.to(self.device)
            source_label = source_label.to(self.device)
            target_data = target_data.to(self.device)

            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)

            optimizer.zero_grad()

            feature_output1, feature_output2, output1, output2 = self.model(source_data, target_data)
            classification_loss = self.loss_class(output1, source_label)
            ddc_loss = self.DDC_loss(feature_output1, feature_output2)
            total_loss = classification_loss + lambda_factor * ddc_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            pred_cls = output1.data.max(1)[1]
            pred.extend(pred_cls.cpu().numpy())
            label.extend(source_label.cpu().numpy())

        label = torch.tensor(label)
        pred = torch.tensor(pred)
            
        f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        f1_score = f1(pred, label).item()
        
        return self.model, classification_loss, f1_score

    def evaluate(self, data_loader):
        self.model.eval()
        
        with torch.no_grad():
            loss_ = 0.0
            acc_ = 0.0
            n_total = 0
            
            pred = []
            target = []
        
            for images, labels in data_loader:
                images = images.to(self.device)
                
                target.extend(labels.numpy())
                labels = labels.to(self.device) 
                size = len(labels)
        
                #data, label = Variable(data), Variable(label)
                _, _, preds, _ = self.model(images, images)
                loss_ += self.loss_class(preds, labels).item()
                pred_cls = preds.data.max(1)[1]
                pred.extend(pred_cls.cpu().numpy())

                acc_ += pred_cls.eq(labels.data).sum().item()
                n_total += size

            loss = loss_ / n_total
            acc = acc_ / n_total
            
            target = torch.tensor(target)
            pred = torch.tensor(pred)
                
            f1 = F1Score(task="multiclass", num_classes=self.num_classes)
            f1_score = f1(pred, target).item()
        
            print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, F1 Score = {f1_score:.2%}")

            return loss, acc, f1_score

'''
############################
         CDAN
############################
'''
class CDANTrainer(BaseTrainer):
    def __init__(self, model, device, class_weights = None, reshape = False):
        super().__init__(model, device, class_weights)
        self.num_classes = self.model.classifier.fc.out_features
        
        loss_class = nn.CrossEntropyLoss(weight=class_weights) 
        self.loss_class = loss_class.to(self.device)
        
        loss_domain = nn.CrossEntropyLoss()
        self.loss_domain = loss_domain.to(self.device)

        self.reshape = reshape

    def train(self, dataloader_source, dataloader_target, optimizer, scheduler, epoch, n_epoch):
        
        self.model.train()
        
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_zip = enumerate(zip(dataloader_source, dataloader_target))

        running_loss = 0
        acc_cls = 0
        loss_class_ = 0
        loss_domain_ = 0
        acc_domain_ = 0
        src_acc_domain_ = 0
        tgt_acc_domain_ = 0
        src_total = 0
        tgt_total = 0
        n_total = 0
        src_preds = []
        src_target = []

        for step, ((images_src, class_src), (images_tgt, _)) in tqdm(data_zip, total=len_dataloader):
            #p = float(step + epoch * len_dataloader) / n_epoch / len_dataloader
            #alpha = 2. / (1. + np.exp(-10 * p)) - 1

            alpha = (epoch+1) / n_epoch
            
            size_src = len(images_src)
            size_tgt = len(images_tgt)


            if self.reshape:
                label_src = torch.zeros(size_src*49).long().to(self.device)  # source 0
                label_tgt = torch.ones(size_tgt*49).long().to(self.device)  # target 1
            else:
                label_src = torch.zeros(size_src).long().to(self.device)  # source 0
                label_tgt = torch.ones(size_tgt).long().to(self.device)  # target 1

            src_target.extend(class_src.numpy())
            class_src = class_src.to(self.device)
            images_src = images_src.to(self.device)
            images_tgt = images_tgt.to(self.device)
        
            self.model.zero_grad()
        
            src_class_output, src_domain_output = self.model(input_data=images_src, alpha=alpha)
            src_preds.extend(src_class_output.detach().cpu().numpy())
            src_loss_class = self.loss_class(src_class_output, class_src) 
            src_loss_domain = self.loss_domain(src_domain_output, label_src)
        
            _, tgt_domain_output = self.model(input_data=images_tgt, alpha=alpha)
            tgt_loss_domain = self.loss_domain(tgt_domain_output, label_tgt)

            total_loss = src_loss_class + (src_loss_domain + tgt_loss_domain)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            src_pred_cls = src_class_output.data.max(1)[1]
            src_pred_domain = src_domain_output.data.max(1)[1]
            tgt_pred_domain = tgt_domain_output.data.max(1)[1]

            loss_class_ += src_loss_class.item() * images_src.size(0)
            loss_domain_ += src_loss_domain.item() * images_src.size(0) + tgt_loss_domain.item() * images_tgt.size(0)
            acc_cls += src_pred_cls.eq(class_src.data).sum().item()
            src_acc_domain_ += src_pred_domain.eq(label_src.data).sum().item()
            tgt_acc_domain_ += tgt_pred_domain.eq(label_tgt.data).sum().item()

            src_total += size_src
            tgt_total += size_tgt
            n_total += (size_src + size_tgt)
        
        src_loss = loss_class_ / src_total
        domain_loss = loss_domain_ / (src_total + tgt_total)
        src_acc = acc_cls / src_total

        if self.reshape:
            src_domain_acc = src_acc_domain_ / (src_total*49)
            tgt_domain_acc = tgt_acc_domain_ / (tgt_total*49)
            domain_acc = (src_acc_domain_ + tgt_acc_domain_) / (n_total*49)
        else:
            src_domain_acc = src_acc_domain_ / (src_total)
            tgt_domain_acc = tgt_acc_domain_ / (tgt_total)
            domain_acc = (src_acc_domain_ + tgt_acc_domain_) / (n_total)
            
        src_target = torch.tensor(src_target)
        src_preds = torch.tensor(np.array(src_preds))
        
        f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        f1_score = f1(src_preds, src_target).item()

        print(f"Training : Avg Loss = {src_loss:.6f}, Avg Accuracy = {src_acc:.2%}, {acc_cls}/{src_total}, F1 Score = {f1_score:.2%}, Avg Domain Accuracy = {domain_acc:.2%}, src = {src_domain_acc:.2%} / tgt = {tgt_domain_acc:.2%}, Avg Domain Loss = {domain_loss:.6f}")
        
        return self.model, src_loss, f1_score


    def evaluate(self, data_loader):
        self.model.eval()
        
        with torch.no_grad():
            loss_ = 0.0
            acc_ = 0.0
            n_total = 0

            pred = []
            target = []
        
            for images, labels in data_loader:
                images = images.to(self.device)
                target.extend(labels.numpy())
                labels = labels.to(self.device) 
                size = len(labels)
                preds, _ = self.model(images, alpha=0)
                loss_ += self.loss_class(preds, labels).item() * images.size(0)
                pred_cls = preds.data.max(1)[1]
                pred.extend(pred_cls.cpu().numpy())
                acc_ += pred_cls.eq(labels.data).sum().item()
                n_total += size
        
            loss = loss_ / n_total
            acc = acc_ / n_total
            
            target = torch.tensor(target)
            pred = torch.tensor(pred)

            f1 = F1Score(task="multiclass", num_classes=self.num_classes)
            f1_score = f1(pred, target).item()
            
            print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, F1 Score = {f1_score:.2%}")

            return loss, acc, f1_score


'''
############################
         DALN
############################
'''
class DALNTrainer(BaseTrainer):
    def __init__(self, model, device, class_weights=None, lambda_nwd=0.1):
        super().__init__(model, device, class_weights)
        self.num_classes = self.model.classifier.fc.out_features
        
        loss_class = nn.CrossEntropyLoss(weight=class_weights) 
        self.loss_class = loss_class.to(self.device)

        self.nwd = NuclearWassersteinDiscrepancy(self.model.classifier).to(self.device)
        self.lambda_nwd = lambda_nwd

    def train(self, dataloader_source, dataloader_target, optimizer, scheduler, epoch, n_epoch):
        self.model.train()
        
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_zip = enumerate(zip(dataloader_source, dataloader_target))

        running_loss = 0
        acc_cls = 0
        loss_class_ = 0
        loss_domain_ = 0
        acc_domain_ = 0
        src_acc_domain_ = 0
        tgt_acc_domain_ = 0
        src_total = 0
        tgt_total = 0
        n_total = 0
        src_preds = []
        src_target = []

        for step, ((images_src, class_src), (images_tgt, _)) in tqdm(data_zip, total=len_dataloader):
        
            size_src = len(images_src)
            size_tgt = len(images_tgt)
            label_src = torch.zeros(size_src).long().to(self.device)  # source 0
            label_tgt = torch.ones(size_tgt).long().to(self.device)  # target 1

            src_target.extend(class_src.numpy())
            class_src = class_src.to(self.device)
            images_src = images_src.to(self.device)
            images_tgt = images_tgt.to(self.device)
        
            self.model.zero_grad()
        
            src_class_output, src_features = self.model(input_data=images_src)
            tgt_class_output, tgt_features  = self.model(input_data=images_tgt)

            
            src_preds.extend(src_class_output.detach().cpu().numpy())
            src_loss_class = self.loss_class(src_class_output, class_src) 
            
            nwd_loss = self.nwd(src_features, tgt_features, 1.0)

            total_loss = src_loss_class + self.lambda_nwd * nwd_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            src_pred_cls = src_class_output.data.max(1)[1]

            loss_class_ += src_loss_class.item() * images_src.size(0)
            loss_domain_ += nwd_loss.item() * (images_src.size(0) + images_tgt.size(0))
            acc_cls += src_pred_cls.eq(class_src.data).sum().item()

            src_total += size_src
            tgt_total += size_tgt
            n_total += (size_src + size_tgt)
        
        src_loss = loss_class_ / src_total
        domain_loss = loss_domain_ / n_total
        src_acc = acc_cls / src_total

        src_target = torch.tensor(src_target)
        src_preds = torch.tensor(np.array(src_preds))
        
        f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        f1_score = f1(src_preds, src_target).item()

        print(f"Training : Avg Loss = {src_loss:.6f}, Avg Accuracy = {src_acc:.2%}, {acc_cls}/{src_total}, F1 Score = {f1_score:.2%}, Avg Domain Loss = {domain_loss:.6f}")
        
        return self.model, src_loss, f1_score


    def evaluate(self, data_loader):
        self.model.eval()
        
        with torch.no_grad():
            loss_ = 0.0
            acc_ = 0.0
            n_total = 0

            pred = []
            target = []
        
            for images, labels in data_loader:
                images = images.to(self.device)
                target.extend(labels.numpy())
                labels = labels.to(self.device) 
                size = len(labels)
                preds, _ = self.model(images)
                loss_ += self.loss_class(preds, labels).item() * images.size(0)
                pred_cls = preds.data.max(1)[1]
                pred.extend(pred_cls.cpu().numpy())
                acc_ += pred_cls.eq(labels.data).sum().item()
                n_total += size
        
            loss = loss_ / n_total
            acc = acc_ / n_total
            
            target = torch.tensor(target)
            pred = torch.tensor(pred)

            f1 = F1Score(task="multiclass", num_classes=self.num_classes)
            f1_score = f1(pred, target).item()
            
            print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, F1 Score = {f1_score:.2%}")

            return loss, acc, f1_score


'''
############################
         Vanila
############################
'''
class VanilaTrainer(BaseTrainer):
    def __init__(self, model, device, class_weights = None, reshape = False):
        super().__init__(model, device, class_weights)
        self.num_classes = self.model.resnet_model.fc.out_features
        
        loss_class = nn.CrossEntropyLoss(weight=class_weights) 
        self.loss_class = loss_class.to(self.device)
        
    def train(self, dataloader_source, _, optimizer, scheduler, epoch, n_epoch):
        
        self.model.train()
        
        len_dataloader = len(dataloader_source)

        running_loss = 0
        acc_cls = 0
        loss_class_ = 0
        src_total = 0
        n_total = 0
        src_preds = []
        src_target = []

        for images_src, class_src in tqdm(dataloader_source, total=len_dataloader):

            size_src = len(images_src)

            src_target.extend(class_src.numpy())
            class_src = class_src.to(self.device)
            images_src = images_src.to(self.device)
            #images_tgt = images_tgt.to(self.device)
        
            self.model.zero_grad()
        
            _, src_class_output= self.model(images_src)
            src_preds.extend(src_class_output.detach().cpu().numpy())
            src_loss_class = self.loss_class(src_class_output, class_src) 

            total_loss = src_loss_class
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            src_pred_cls = src_class_output.data.max(1)[1]
            loss_class_ += src_loss_class.item() * images_src.size(0)
            acc_cls += src_pred_cls.eq(class_src.data).sum().item()

            src_total += size_src
            n_total += size_src
        
        src_loss = loss_class_ / src_total
        src_acc = acc_cls / src_total


        src_target = torch.tensor(src_target)
        src_preds = torch.tensor(np.array(src_preds))
        
        f1 = F1Score(task="multiclass", num_classes=self.num_classes)
        f1_score = f1(src_preds, src_target).item()

        print(f"Training : Avg Loss = {src_loss:.6f}, Avg Accuracy = {src_acc:.2%}, {acc_cls}/{src_total}, F1 Score = {f1_score:.2%}")
        
        return self.model, src_loss, f1_score


    def evaluate(self, data_loader):
        self.model.eval()
        
        with torch.no_grad():
            loss_ = 0.0
            acc_ = 0.0
            n_total = 0

            pred = []
            target = []
        
            for images, labels in data_loader:
                images = images.to(self.device)
                target.extend(labels.numpy())
                labels = labels.to(self.device) 
                size = len(labels)
                _, preds = self.model(images)
                loss_ += self.loss_class(preds, labels).item() * images.size(0)
                pred_cls = preds.data.max(1)[1]
                pred.extend(pred_cls.cpu().numpy())
                acc_ += pred_cls.eq(labels.data).sum().item()
                n_total += size
        
            loss = loss_ / n_total
            acc = acc_ / n_total
            
            target = torch.tensor(target)
            pred = torch.tensor(pred)

            f1 = F1Score(task="multiclass", num_classes=self.num_classes)
            f1_score = f1(pred, target).item()
            
            print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, F1 Score = {f1_score:.2%}")

            return loss, acc, f1_score


'''
############################
        Adamatch
############################
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import F1Score


class AdamatchTrainer:
    def __init__(self, encoder, classifier, device, hyperparams, class_weights=None):
        self.device = device
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.hyperparams = hyperparams

        # 기본값: 1.0
        self.lambda_u = hyperparams.get("lambda_u", 1.0)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

        self.history = {
            'epoch_loss': [],
            'accuracy_source': [],
            'accuracy_target': []
        }

    def train(self,
              dataloader_src_weak, dataloader_src_strong,
              dataloader_tgt_weak, dataloader_tgt_strong,
              optimizer, scheduler,
              epoch, n_epoch,
              current_step=0,
              lambda_u=None):   # [수정] lambda_u 인자 추가

        self.encoder.train()
        self.classifier.train()

        tau = self.hyperparams.get("tau", 0.9)
        iters = min(len(dataloader_src_weak), len(dataloader_src_strong),
                    len(dataloader_tgt_weak), len(dataloader_tgt_strong))
        steps_per_epoch = iters
        total_steps = n_epoch * steps_per_epoch

        running_loss, running_src_loss, running_tgt_loss = 0.0, 0.0, 0.0

        dataset = zip(dataloader_src_weak, dataloader_src_strong,
                      dataloader_tgt_weak, dataloader_tgt_strong)

        for (src_w, labels_src), (src_s, _), (tgt_w, _), (tgt_s, _) in dataset:
            src_w, labels_src = src_w.to(self.device), labels_src.to(self.device)
            src_s = src_s.to(self.device)
            tgt_w = tgt_w.to(self.device)
            tgt_s = tgt_s.to(self.device)

            # concat
            data_combined = torch.cat([src_w, src_s, tgt_w, tgt_s], 0)
            source_combined = torch.cat([src_w, src_s], 0)
            source_total = source_combined.size(0)

            optimizer.zero_grad()

            # forward pass 1
            logits_combined = self.classifier(self.encoder(data_combined))
            logits_source_p = logits_combined[:source_total]

            # forward pass 2 (BN off)
            self._disable_batchnorm_tracking(self.encoder)
            self._disable_batchnorm_tracking(self.classifier)
            logits_source_pp = self.classifier(self.encoder(source_combined))
            self._enable_batchnorm_tracking(self.encoder)
            self._enable_batchnorm_tracking(self.classifier)

            # logit interpolation
            lambd = torch.rand_like(logits_source_p).to(self.device)
            final_logits_source = (lambd * logits_source_p) + ((1 - lambd) * logits_source_pp)

            # distribution alignment
            logits_source_weak = final_logits_source[:src_w.size(0)]
            pseudolabels_source = F.softmax(logits_source_weak, 1)

            logits_target = logits_combined[source_total:]
            logits_target_weak = logits_target[:tgt_w.size(0)]
            pseudolabels_target = F.softmax(logits_target_weak, 1)

            # 클래스별 평균 확률 비율 계산
            ratio = (1e-6 + torch.mean(pseudolabels_source, dim=0)) / (1e-6 + torch.mean(pseudolabels_target, dim=0))
            adjusted = pseudolabels_target * ratio
            final_pseudolabels = adjusted / adjusted.sum(dim=1, keepdim=True)

            # relative confidence thresholding
            row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
            final_sum = torch.mean(row_wise_max, 0)
            c_tau = tau * final_sum

            max_values, _ = torch.max(final_pseudolabels, dim=1)
            mask = (max_values >= c_tau).float()

            # losses
            source_loss = self._compute_source_loss(
                logits_source_weak,
                final_logits_source[src_w.size(0):],
                labels_src
            )

            final_pseudolabels = torch.max(final_pseudolabels, 1)[1]
            target_loss = self._compute_target_loss(
                final_pseudolabels,
                logits_target[tgt_w.size(0):],
                mask
            )

            # === μ schedule (cosine ramp-up) ===
            progress = current_step / total_steps
            # [수정] 외부에서 들어온 lambda_u 사용
            base_lambda = lambda_u if lambda_u is not None else self.lambda_u
            mu = base_lambda * 0.5 * (1 - np.cos(np.pi * progress))

            loss = source_loss + (mu * target_loss)
            current_step += 1

            # backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_src_loss += source_loss.item()
            running_tgt_loss += target_loss.item()

        # epoch 결과
        epoch_loss = running_loss / iters
        src_loss = running_src_loss / iters
        tgt_loss = running_tgt_loss / iters

        src_loss_eval, src_acc, _ = self.evaluate(dataloader_src_weak)
        tgt_loss_eval, tgt_acc, _ = self.evaluate(dataloader_tgt_weak)

        self.history['epoch_loss'].append(epoch_loss)
        self.history['accuracy_source'].append(src_acc)
        self.history['accuracy_target'].append(tgt_acc)

        print(f"[Epoch {epoch+1}/{n_epoch}] "
              f"loss={epoch_loss:.6f}; src_loss={src_loss:.6f}; tgt_loss={tgt_loss:.6f}; "
              f"src_acc={src_acc:.4f}; tgt_acc={tgt_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

        return self.encoder, self.classifier, epoch_loss, src_loss, tgt_loss, src_acc, tgt_acc


    def evaluate(self, data_loader):
        self.encoder.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            loss_ = 0.0
            acc_ = 0.0
            n_total = 0
    
            pred = []
            target = []
        
            for images, labels in data_loader:
                images = images.to(self.device)
                target.extend(labels.numpy())
                labels = labels.to(self.device) 
                size = len(labels)
    
                preds = self.classifier(self.encoder(images))
                loss_ += self.loss_fn(preds, labels).item() * images.size(0)
                pred_cls = preds.data.max(1)[1]
                pred.extend(pred_cls.cpu().numpy())
                acc_ += pred_cls.eq(labels.data).sum().item()
                n_total += size
        
            loss = loss_ / n_total
            acc = acc_ / n_total
            
            target = torch.tensor(target)
            pred = torch.tensor(pred)
    
            f1 = F1Score(task="multiclass", num_classes=preds.size(1)).to(self.device)
            f1_score = f1(pred, target).item()
            
            print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, F1 Score = {f1_score:.2%}")
    
            return loss, acc, f1_score


    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False
        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True
        model.apply(fn)

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels):
        loss_function = nn.CrossEntropyLoss()
        weak_loss = loss_function(logits_weak, labels)
        strong_loss = loss_function(logits_strong, labels)
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach()
        loss = loss_function(logits_strong, pseudolabels)
        return (loss * mask).mean()
