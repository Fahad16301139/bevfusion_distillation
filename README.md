# BEVFusion Distillation Training Workflow

When you run the training script (final_working_distillation.py), here's the execution flow:

## 1. Initialization Phase

```python
# First, the main function is called
def main():
    # Parse arguments and setup

    # Create the distillation trainer
    trainer = FinalDistillationTrainer(
        teacher_config_path=args.teacher_config,
        student_config_path=args.student_config,
        teacher_checkpoint=args.teacher_checkpoint
    )
```

## 2. Trainer Initialization

```python
# Inside FinalDistillationTrainer.__init__
# INTEGRATION POINT 1: BEVFusion models are loaded and wrapped
self.teacher_wrapper, self.student_wrapper, self.distill_criterion = build_bevfusion_distillation(
    teacher_config_path=teacher_config_path,
    student_config_path=student_config_path,
    teacher_checkpoint=teacher_checkpoint,
    n_data=1000,
    device=self.device
)
```

## 3. Model Building (in build_bevfusion_distillation)

```python
# Load configs and build BEVFusion models
teacher_config = Config.fromfile(teacher_config_path)
student_config = Config.fromfile(student_config_path)

# INTEGRATION POINT 2: BEVFusion teacher model created
teacher_model = MODELS.build(teacher_config.model)
# Load teacher checkpoint and freeze weights

# INTEGRATION POINT 3: BEVFusion student model created  
student_model = MODELS.build(student_config.model)

# INTEGRATION POINT 4: BEVFusion models are wrapped for feature extraction
teacher_wrapper = BEVFusionDistillationWrapper(teacher_model, is_student=False)
student_wrapper = BEVFusionDistillationWrapper(student_model, is_student=True)

# INTEGRATION POINT 5: Create distillation loss for BEVFusion features
distill_criterion = BEVFusionDistillationLoss(
    student_channels=student_channels,
    teacher_channels=teacher_channels,
    # CRD parameters...
)
```

## 4. Training Loop

```python
# Loop through dataset and epochs
for epoch in range(start_epoch, num_epochs):
    for batch_idx, data_batch in enumerate(dataloader):
        # Process each batch with train_step
        losses = trainer.train_step(data_batch)
```

## 5. Training Step (Single Batch)

```python
# Inside train_step method of FinalDistillationTrainer
def train_step(self, data_batch):
    # Validate and fix data format
    data_batch = self.validate_and_fix_data(data_batch)
    
    # INTEGRATION POINT 6: Get features from BEVFusion teacher model
    with torch.no_grad():
        teacher_loss, teacher_features = self.teacher_wrapper(batch_inputs_dict, batch_data_samples, mode='loss')
    
    # INTEGRATION POINT 7: Get features from BEVFusion student model
    student_loss, student_features = self.student_wrapper(batch_inputs_dict, batch_data_samples, mode='loss')
    
    # Calculate task loss from student model's output
    task_loss = sum(v for k, v in student_loss.items() if 'loss' in k)
    
    # INTEGRATION POINT 8: Compute CRD distillation loss using BEVFusion features
    indices = torch.randint(0, 1000, (batch_size,), device=self.device)
    distill_losses = self.distill_criterion(student_features, teacher_features, indices)
    distill_loss = distill_losses.get('total_distill', torch.tensor(0.0, device=self.device))
    
    # Combine losses and backpropagate
    total_loss = task_loss + alpha_distill * distill_loss
    total_loss.backward()
    
    # Update student model parameters
    self.optimizer.step()
```

## 6. Under the Hood: Feature Extraction

During both teacher_wrapper and student_wrapper forward passes:

```python
# In BEVFusionDistillationWrapper.forward
# The underlying BEVFusion model is called
if mode == 'loss':
    result = self.model.loss(batch_inputs_dict, batch_data_samples)
else:
    result = self.model.predict(batch_inputs_dict, batch_data_samples)

# Meanwhile, registered hooks capture features during the forward pass
def _get_activation_hook(self, name):
    def hook(module, input, output):
        # Store output features in self.features dictionary
        self.features[name] = output
```

## 7. Under the Hood: Distillation Loss Computation

When the distill_criterion is called:

```python
# In BEVFusionDistillationLoss.forward
for layer_name in student_features.keys():
    if layer_name in teacher_features and layer_name in self.crd_losses:
        # Get features for this layer from both models
        s_feat = student_features[layer_name]  # From student BEVFusion
        t_feat = teacher_features[layer_name]  # From teacher BEVFusion
        
        # INTEGRATION POINT 9: Call CRD loss for this specific layer's features
        crd_loss = self.crd_losses[layer_name](s_feat, t_feat, indices)
        weighted_loss = self.alpha_crd * crd_loss
        losses[f'{layer_name}_crd'] = weighted_loss
        total_loss += weighted_loss
```

## 8. Under the Hood: CRD Loss Computation

Inside each CRDLoss module's forward call:

```python
# In CRDLoss.forward
# Project features to embedding space
f_s = self.embed_s(f_s)  # Student features
f_t = self.embed_t(f_t)  # Teacher features

# Compute contrastive loss using memory bank
out_s, out_t = self.contrast(f_s, f_t, idx)
s_loss = self.criterion_s(out_s)
t_loss = self.criterion_t(out_t)
loss = s_loss + t_loss
```

## 9. Model Saving & Evaluation

```python
# After certain epochs or batches
if epoch % save_interval == 0:
    # Save model checkpoint
    ckpt_path = os.path.join(work_dir, f'epoch_{epoch}.pth')
    save_checkpoint(...)
    
    # Optionally evaluate model
    # ...
```

This workflow shows how BEVFusion models are wrapped, how features are extracted during forward passes, and how these features are fed into the CRD distillation framework to calculate the distillation loss that guides the student's training.
