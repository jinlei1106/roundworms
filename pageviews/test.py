from global_decorator import SelfTemplateView


class TestPageView(SelfTemplateView):
    template_name = 'test.html'
