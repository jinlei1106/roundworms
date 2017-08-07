from django.views.generic import TemplateView


class TestPageView(TemplateView):
    template_name = 'test.html'
