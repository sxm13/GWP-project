from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS

from skopt.space import Real, Integer, Categorical

from skopt.utils import use_named_args
from skopt import gp_minimize

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

import pandas as pd
import numpy as np
import csv
import joblib
import shap
import joblib
import matplotlib.pyplot as plt
    
import warnings
from shap.plots._labels import labels
from shap.utils import format_value, ordinal_str,safe_isinstance, OpChain, format_value
from shap.plots._utils import convert_ordering, convert_color, merge_nodes, get_sort_order, sort_inds, dendrogram_coords
from shap.plots import colors

import scipy
import copy
from shap import Explanation, Cohorts

def load_data(csvfile, splitfile,diverse_ratio = 0.8):

    remaining_ratio = 1 - diverse_ratio
    diverse_set=[]
    remaining_set=[]
    
    df = pd.read_csv(csvfile)
    N_samples = df.shape[0]
    N_features = df.shape[1] - 4
    N_targets = 2
    
    data_file_name = csvfile
    
    txt = open(splitfile,'r').read()
    print(" Load file name : " + splitfile)
    s1=txt.find("[",0)
    s2=txt.find("]",s1)
    diverse_set=txt[s1+1:s2].split(", ")
    diverse_set=[int(i) for i in diverse_set]
    s3=txt.find("[",s2)
    s4=txt.find("]",s3)
    remaining_set=txt[s3+1:s4].split(", ")
    remaining_set=[int(i) for i in remaining_set]
    print("# of diverse set, remaining set :",len(diverse_set),len(remaining_set))

    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        number = np.empty((N_samples,))
        structure = np.empty((N_samples,))
        data = np.empty((N_samples, N_features))
        target = np.empty((N_samples,N_targets))
        structure = []
        feature_names=temp[2:2+N_features]
        for i, d in enumerate(data_file):
            number[i] = np.asarray(d[0],dtype=np.int)
            structure.append(d[1])
            data[i] = np.asarray(d[2:2+N_features], dtype=np.float64)
            target[i] = np.asarray(d[-N_targets:], dtype=np.float64)
    N_materials = data.shape[0]

    diverse_set_total=[]
    remaining_set_total=[]
    for i,diverse in enumerate(diverse_set):
        arridx = np.where(number == diverse)
        for j,element_div in enumerate(arridx[0]):
            diverse_set_total.append(element_div)
    for i,remaining in enumerate(remaining_set):
        arridx = np.where(number == remaining)
        for j,element_rem in enumerate(arridx[0]):
            remaining_set_total.append(element_rem)        

    X_train = data[diverse_set_total]
    y_train = target[diverse_set_total]
    X_test = data[remaining_set_total]
    y_test = target[remaining_set_total]

    return X_train, y_train, X_test, y_test

def training_models(Xtrain, Ytrain, Xtest, Ytest, model_name, n_job = 4):

    reg = XGBRegressor()
    space  = [Integer(1,200, name='n_estimators'),
            Integer(1, 10, name='max_depth'),
            Integer(1, 10, name='num_parallel_tree'),
            Integer(1, 10, name='min_child_weight'),
            Real(0.001,1,"log-uniform",name='learning_rate'),
            Real(0.01,1,name='subsample'),
            Real(0.001,10,"log-uniform",name='gamma'),
            Real(0, 1, name='alpha'),
            Real(2, 10, name='reg_alpha'),
            Real(10, 50, name='reg_lambda')
         ]
    @use_named_args(space)

    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
    
        print(result)
        return result

    res_gp = gp_minimize(objective, space, n_calls=100)

    print("Best score=%.4f" % res_gp.fun)
    print("""Best parameters:
        - n_estimator=%d
        - max_depth=%d
        - num_parallel_tree=%d
        - min_child_weight=%d
        - learning_rate=%f
        - subsample=%f
        - gamma=%f
        - alpha=%f
        - reg_alpha=%f
        - reg_lambda=%f""" % (res_gp.x[0],res_gp.x[1],
                            res_gp.x[2],res_gp.x[3],
                            res_gp.x[4],res_gp.x[5],
                            res_gp.x[6],res_gp.x[7],
                            res_gp.x[8],res_gp.x[9]
                             ))
    reg_opt = XGBRegressor(n_estimators=res_gp.x[0],
                            max_depth=res_gp.x[1],
                            num_parallel_tree=res_gp.x[2],
                            min_child_weight=res_gp.x[3],
                            learning_rate=res_gp.x[4],
                            subsample=res_gp.x[5],
                            gamma=res_gp.x[6],
                            alpha=res_gp.x[7],
                            reg_alpha=res_gp.x[8],
                            reg_lambda=res_gp.x[9]
                            )
    reg_opt.fit(Xtrain, Ytrain)

    print('R^2 Training Score: {:.3f} \nR^2 Testing Score: {:.3f}'.format(reg_opt.score(Xtrain, Ytrain),reg_opt.score(Xtest, Ytest)))
    print('MAE Training Score: {:.3f} \nMAE Testing Score: {:.3f}'.format(mean_absolute_error(Ytrain,reg_opt.predict(Xtrain)),mean_absolute_error(Ytest,reg_opt.predict(Xtest))))
    print('RMSE Training Score: {:.3f} \nRMSE Testing Score: {:.3f}'.format(np.sqrt(mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)),np.sqrt(mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False))))

    save_model = joblib.dump(reg_opt,model_name+".pkl")

def features_analysis(model,csvfile, max_display=10, order=Explanation.abs.mean(0),
             clustering=None, cluster_threshold=0.5, color=None, axis_color="#333333", alpha=1, show=True, log_scale=False,
             color_bar=True, plot_size="auto", color_bar_label=labels["FEATURE_VALUE"]):
    data = pd.read_csv(csvfile)
    X = pd.DataFrame(data,columns = ['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT'])
    
    ML_model = joblib.load(model)
    explainer = shap.Explainer(ML_model)
    shap_values = explainer(X)
    feature = ['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT']

    fig = plt.figure(figsize=(6.5,5),dpi=600)
    
    if not isinstance(shap_values, Explanation):
        raise ValueError("the beeswarm plot requires Explanation object as the `shap_values` argument")

    if len(shap_values.shape) == 1:
        raise ValueError(
            "The beeswarm plot does not support plotting a single instance, please pass "
            "an explanation matrix with many instances!"
        )
    elif len(shap_values.shape) > 2:
        raise ValueError(
            "The beeswarm plot does not support plotting explanations with instances that have more "
            "than one dimension!"
        )
    shap_exp = shap_values

    values = np.copy(shap_exp.values)
    features = shap_exp.data
        
    feature_names = feature

    order = convert_ordering(order, values)
    
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = values.shape[1]

    if features is not None:
        shape_msg = "The shape of the matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if log_scale:
        plt.xscale('symlog')

    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
        if partition_tree is not None and partition_tree.var(0).sum() == 0:
            partition_tree = partition_tree[0]
        else:
            partition_tree = None
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering
    
    if partition_tree is not None:
        assert partition_tree.shape[1] == 4, "The clustering provided by the Explanation object does not seem to be a partition tree (which is all shap.plots.bar supports)!"

    # plotting SHAP interaction values
    if len(values.shape) == 3:

        if plot_type == "compact_dot":
            new_values = values.reshape(values.shape[0], -1)
            new_features = np.tile(features, (1, 1, features.shape[1])).reshape(features.shape[0], -1)

            new_feature_names = []
            for c1 in feature_names:
                for c2 in feature_names:
                    if c1 == c2:
                        new_feature_names.append(c1)
                    else:
                        new_feature_names.append(c1 + "* - " + c2)

            return beeswarm(
                new_values, new_features, new_feature_names,
                max_display=max_display, plot_type="dot", color=color, axis_color=axis_color,
                title=title, alpha=alpha, show=show, sort=sort,
                color_bar=color_bar, plot_size=plot_size, class_names=class_names,
                color_bar_label="*" + color_bar_label
            )

        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        interaction_sort_inds = order#np.argsort(-np.abs(values.sum(1)).sum(0))

        # get plotting limits
        delta = 1.0 / (values.shape[1] ** 2)
        slow = np.nanpercentile(values, delta)
        shigh = np.nanpercentile(values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        plt.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
        plt.subplot(1, max_display, 1)
        proj_values = values[:, interaction_sort_inds[0], interaction_sort_inds]
        proj_values[:, 1:] *= 2  # because off diag effects are split in half
        beeswarm(
            proj_values, features[:, interaction_sort_inds] if features is not None else None,
            feature_names=feature_names[interaction_sort_inds],
            sort=False, show=False, color_bar=False,
            plot_size=None,
            max_display=max_display
        )
        plt.xlim((slow, shigh))
        plt.xlabel("")
        title_length_limit = 11
        plt.title(shorten_text(feature_names[interaction_sort_inds[0]], title_length_limit))
        for i in range(1, min(len(interaction_sort_inds), max_display)):
            ind = interaction_sort_inds[i]
            plt.subplot(1, max_display, i + 1)
            proj_values = values[:, ind, interaction_sort_inds]
            proj_values *= 2
            proj_values[:, i] /= 2  # because only off diag effects are split in half
            summary(
                proj_values, features[:, interaction_sort_inds] if features is not None else None,
                sort=False,
                feature_names=["" for i in range(len(feature_names))],
                show=False,
                color_bar=False,
                plot_size=None,
                max_display=max_display
            )
            plt.xlim((slow, shigh))
            plt.xlabel("")
            if i == min(len(interaction_sort_inds), max_display) // 2:
                plt.xlabel(labels['INTERACTION_VALUE'])
            plt.title(shorten_text(feature_names[ind], title_length_limit))
        plt.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        plt.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            plt.show()
        return

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()
    while True:
        feature_order = convert_ordering(order, Explanation(np.abs(values)))
        if partition_tree is not None:

            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values))

            # now relax the requirement to match the parition tree ordering for connections above cluster_threshold
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, cluster_threshold, feature_order)
        
            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= cluster_threshold:
                #values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for i in range(len(values)):
                    values[:,ind1] += values[:,ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
        values[:,feature_order[num_features-1]] = np.sum([values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)
    
    # build our y-tick labels
    yticklabels = [feature_names[i] for i in feature_inds]
    if num_features < len(values[0]):
        yticklabels[-1] = "Sum of %d other features" % num_cut
    
    row_height = 0.4
    plt.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]
        fvalues = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if fvalues is not None:
            fvalues = fvalues[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: # check categorical feature
                colored_feature = False
            else:
                fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
        except:
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if safe_isinstance(color, "matplotlib.colors.Colormap") and features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax

            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)
            plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                        vmax=vmax, s=16, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                        cmap=color, vmin=vmin, vmax=vmax, s=16,
                        c=cvals, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)
        else:

            plt.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                        color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)


    if safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = plt.colorbar(m, ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, fontdict={'family':'Calibri','size':22})
        cb.ax.tick_params(labelsize=16, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(range(len(feature_inds)), reversed(yticklabels), fontsize=16)
    plt.ylim(-1, len(feature_inds))
    plt.xlabel("SHAP value", fontdict={'family':'Calibri','size':22})
    
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.tick_params(axis='both', which='both', direction='out',width=1.5,labelsize=16)
    
    plt.text(20,-0.7,"d",fontdict={'family':'Calibri','size':24,'fontweight':'bold'})
    
    if show:
        plt.show()
