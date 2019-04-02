from django.views.generic import TemplateView


class SelfTemplateView(TemplateView):
    def get_context_data(self, **kwargs):
        user_agent = self.request.META['HTTP_USER_AGENT'].lower() if 'HTTP_USER_AGENT' in self.request.META else ''
        base_html = 'base/pc_base.html'
        for key in ('ipad', 'iphone', 'ipod', 'android'):
            if key in user_agent:
                base_html = 'base/mobile_base.html'
                break
        kwargs['basehtml'] = base_html
        if 'view' not in kwargs:
            kwargs['view'] = self
        return kwargs
