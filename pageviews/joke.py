from global_decorator import SelfTemplateView


class JokePageView(SelfTemplateView):
    template_name = 'joke.html'
