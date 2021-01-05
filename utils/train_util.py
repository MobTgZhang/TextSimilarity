import logging
import logging
logger = logging.getLogger()
import utils.tool as tool
# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats,train_saver):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = tool.AverageMeter()
    epoch_time = tool.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(model.update(ex))
        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.4f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)