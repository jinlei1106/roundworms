from django.views.generic import TemplateView


class JokePageView(TemplateView):
    template_name = 'joke.html'
