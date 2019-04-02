from global_decorator import SelfTemplateView


class HomePageView(SelfTemplateView):
    template_name = 'home.html'
