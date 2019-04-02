import random
from global_decorator import SelfTemplateView

from utils.docs import explanation_dict


class ResultPageView(SelfTemplateView):
    template_name = 'result.html'

    def get_context_data(self, **kwargs):
        context = super(ResultPageView, self).get_context_data(**kwargs)
        mod_lst = ['高兴', '愤怒', '恐惧', '忧愁', '痛苦', '紧张', '激动']
        index = random.randint(0, len(mod_lst)-1)
        title = mod_lst[index]
        context.update({'title': title, 'explanation': explanation_dict[title]})
        return context
