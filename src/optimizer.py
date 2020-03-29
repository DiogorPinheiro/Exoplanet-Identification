

def triage(model, experiment, train_X_global, train_Y_global, val_X_global, val_Y_global, test_X_global, test_Y_global, epoch, batch_size, lay1_filters, l1_kernel_size, pool_size, strides, conv_dropout, lay2_filters, l2_kernel_size, dense_f, dense_dropout, x_train_global, train_X_local, train_Y_local, val_X_local, val_Y_local, test_X_local, test_Y_local, x_train_local):
    #model = seqModelCNN(lay1_filters,l1_kernel_size,pool_size,strides,conv_dropout,lay2_filters,l2_kernel_size,dense_f,dense_dropout,x_train_global)
    model = bothViewsCNN(train_X_global, train_X_local, lay1_filters, l1_kernel_size, pool_size,
                         strides, conv_dropout, lay2_filters, l2_kernel_size, dense_f, dense_dropout)

    # Local or Global View
    model.fit(train_X_global, train_Y_global, batch_size=batch_size, epochs=epoch, validation_data=(val_X_global,
                                                                                                    val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    score = model.evaluate(test_X_global, test_Y_global, verbose=0)[1]

    # Local and Global View
    #model.fit([train_X_global,train_X_local], train_Y_global, batch_size=batch_size, epochs=epoch,validation_data=([val_X_global,val_X_local], val_Y_global), callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    #score = model.evaluate([test_X_global,test_X_local], test_Y_global, verbose=0)[1]

    return score


def mainOptimizer():
