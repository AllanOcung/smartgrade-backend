# Generated migration for DataUploadBatch uploaded_by field fix
from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        ('student_data', '0002_alter_studentrecord_bsm1201_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datauploadbatch',
            name='uploaded_by',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
    ]
