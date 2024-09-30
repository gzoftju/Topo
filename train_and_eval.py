import numpy as np
import copy
import torch
import dgl
from utils import set_seed

"""
1. Train and eval
"""


def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    _, logits, loss, dist, codebooklogits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss += criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val


def train_sage(model, dataloader, feats, labels, criterion, optimizer,g):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]
        # Compute loss and prediction
        _, logits, loss , _ , _,perplexity,_= model(blocks, batch_feats,g)
        out = logits.log_softmax(dim=1)
        # print(loss)
        #loss += criterion(out, batch_labels)
        # print(loss)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        _, logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels[idx_batch[i]])
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


"""
Train student MLP model with Token-Based GNN-MLP Distillation.
"""


def train_mini_batch_token(model, feats, codebook_embeddings, tea_soft_token_assignments_all, batch_size, criterion,
                           optimizer, lamb=1, temperature=4):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        # No graph needed for the forward function
        h_list, _ = model(None, feats[idx_batch[i]])
        tea_soft_token_assignments = tea_soft_token_assignments_all[idx_batch[i]]

        # Compute student soft token assignments by calculating the L2 distance between student features and teacher codebook embeddings.
        stu_soft_token_assignments = - torch.cdist(h_list[-1], codebook_embeddings, p=2)
        tea_soft_token_assignments = tea_soft_token_assignments / temperature
        stu_soft_token_assignments = stu_soft_token_assignments / temperature
        tea_soft_token_assignments = tea_soft_token_assignments.softmax(dim=-1)
        tea_soft_token_assignments = torch.squeeze(tea_soft_token_assignments)
        stu_soft_token_assignments = stu_soft_token_assignments.log_softmax(dim=-1)
        stu_soft_token_assignments = torch.squeeze(stu_soft_token_assignments)

        # Compare student and teacher soft token assignments by KL divergence.
        loss = criterion(stu_soft_token_assignments, tea_soft_token_assignments) * temperature * temperature
        loss *= lamb
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def evaluate(model, data, feats, labels, criterion, evaluator, g):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        h_list, logits, loss , dist_node,dist_link, codebook , total_perplexity ,gumbel_emb_node,gumbel_emb_link= model.inference(data, feats,g)
        out = logits.log_softmax(dim=1)
    return out, loss, 0, h_list, dist_node,dist_link, codebook , total_perplexity,gumbel_emb_node,gumbel_emb_link


def evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator, idx_eval=None):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            _, logits = model.inference(None, feats[batch_size * i: batch_size * (i + 1)])
            out = logits.log_softmax(dim=1)
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return out_all, loss.item(), score


"""
2. Run teacher
"""

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def run_transductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
            worker_init_fn=worker_init_fn,
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighbors.
        sampler_eval = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes())[idx_val],
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
            worker_init_fn=worker_init_fn,
        )
        dataloader_test = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes())[idx_test],
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
            worker_init_fn=worker_init_fn,
        )
        data = dataloader
        data_eval = dataloader_eval
        data_test = dataloader_test
    elif "MLP" in model.model_name:
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
    else:
        g = g.to(device)
        data = g
        data_eval = g
    
    best_epoch, best_score_val, count, best_loss_val,best_total_perplexity= 0, 0, 0,100,0
    #CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    for epoch in range(1, conf["max_epoch"] + 1):
        print(epoch)
        if "SAGE" in model.model_name:
            loss = train_sage(model, data, feats, labels, criterion, optimizer,g)
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train)
        #CosineLR.step()
        if epoch % conf["eval_interval"] == 0:
              out, loss_train, score_train,  h_list, dist_node,dist_link, codebook,val_total_perplexity ,gumbel_emb_node,gumbel_emb_link= evaluate(
                    model, data_eval, feats, labels, criterion, evaluator,g)
              ______, loss_test, _,  __, ___,____, _____ ,test_total_perplexity,gumbel_emb_node,gumbel_emb_link= evaluate(
                    model, data_test, feats, labels, criterion, evaluator,g)
              logger.info(
                  f"Ep {epoch:3d} | loss_train: {loss:.4f} | loss_val: {loss_train:.4f} | loss_test: {loss_test:.4f} | val_total_perplexity: {val_total_perplexity:.4f} | test_total_perplexity: {test_total_perplexity:.4f}"
              )
              out=out.cpu()
              loss_train=loss_train
              score_train=score_train
              h_list=h_list
              dist_node,dist_link, codebook=dist_node.cpu(),dist_link.cpu(), codebook.cpu()
              gumbel_emb_node,gumbel_emb_link=gumbel_emb_node.cpu(),gumbel_emb_link.cpu()
              torch.cuda.empty_cache()
              
              
              loss_val = criterion(out[idx_val], labels[idx_val].cpu()).item()
              score_val = evaluator(out[idx_val], labels[idx_val].cpu())
              loss_test = criterion(out[idx_test], labels[idx_test].cpu()).item()
              acc = evaluator(out[idx_test], labels[idx_test].cpu())
              


              loss_and_score += [
                  [
                      epoch,
                      loss_train,
                      loss_val,
                      loss_test,
                      score_train,
                      score_val,
                      acc,
                  ]
              ]
  
              if loss_train <= best_loss_val:
                  best_epoch = epoch
                  best_loss_val = loss_train
                  state = copy.deepcopy(model.state_dict())
                  count = 0
              else:
                  count += 1
    model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_val
        )
    else:
        out, _, score_val, h_list, dist_node,dist_link, codebook,val_total_perplexity,gumbel_emb_node,gumbel_emb_link = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, g)

    ______, loss_test, _,  __, ___,____, _____ ,test_total_perplexity,gumbel_emb_node,gumbel_emb_link= evaluate(model, data_test, feats, labels, criterion, evaluator,g)
    
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, loss: {loss_test :.4f} , val_total_perplexity: {val_total_perplexity:.4f} , test_total_perplexity: {test_total_perplexity:.4f}"
    )
    out, loss, score_train,  h_list, dist_node,dist_link, codebook,total_perplexity,gumbel_emb_node,gumbel_emb_link = evaluate(
                    model, data, feats, labels, criterion, evaluator,g)
                    
    return out, score_val, acc, h_list, dist_node,dist_link, codebook,gumbel_emb_node,gumbel_emb_link


def run_inductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader = dgl.dataloading.NodeDataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.NodeDataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    best_epoch, best_score_val, count ,best_loss_val= 0, 0, 0 , 1
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(
                model, obs_data, obs_feats, obs_labels, criterion, optimizer
            )
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                _, loss_test_tran, acc_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    criterion,
                    batch_size,
                    evaluator,
                )
                _, loss_test_ind, acc_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    criterion,
                    batch_size,
                    evaluator,
                )
            else:
                obs_out, loss_train, score_train, h_list, dist, codebook = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                loss_val = criterion(
                    obs_out[obs_idx_val], obs_labels[obs_idx_val]
                ).item()
                score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                loss_test_tran = criterion(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                ).item()
                acc_tran = evaluator(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                )

                # Evaluate the inductive part with the full graph
                out, loss_test_ind, acc_ind, h_list, dist, codebook = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
                )
            logger.info(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {acc_tran:.4f} | s_ti: {acc_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    acc_tran,
                    acc_ind,
                ]
            ]
            if loss_val <= best_loss_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
        )
        out, _, acc_ind = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
        )

    else:
        obs_out, _, score_val, h_list, dist, codebook = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_labels,
            criterion,
            evaluator,
            obs_idx_val,
        )
        out, _, acc_ind, h_list, dist, codebook = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
        )

    acc_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, acc_tran: {acc_tran :.4f}, acc_ind: {acc_ind :.4f}"
    )
    
    
    return out, score_val, acc_tran, acc_ind, h_list, dist, codebook

